"""
IMPORTANT WARNING: Modification of this code is prohibited, any alterations will be detected.
v1.0.0
"""
import atexit
import multiprocessing
from pathlib import Path
import sys
import psutil
import os
import time
import signal
import hashlib
from audit_utils import secure_logger


logger = secure_logger.get_logger('monitor')
version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
logger.info(f"monitor v1.0.0 {version}")


class UIDResourceMonitor:
    def __init__(self):
        self.current_uid = os.getuid()
        
        # 记录基线 (Baseline): 监控启动瞬间，所有存活进程已经发生过的历史 I/O
        # 键: (pid, create_time)  值: {'bytes': 0, 'chars': 0}
        self.baseline_io = {}
        
        # 记录最高水位线: 监控期间观察到的每个进程的最大 I/O
        self.max_seen_io = {}

        self._record_baseline()

    def _get_process_io_data(self):
        """
        内部方法：抓取当前时刻 UID 名下所有进程的 IO 状态
        """
        current_data = {}
        total_rss = 0
        total_vms = 0
        
        # 增加 'create_time' 用于防止 PID 复用导致的统计错乱
        attrs = ['uids', 'memory_info', 'io_counters', 'create_time']
        
        for p in psutil.process_iter(attrs):
            try:
                uids = p.info.get('uids')
                if uids and uids.real == self.current_uid:
                    
                    # 累加内存（内存是瞬时状态，直接算总和）
                    mem = p.info.get('memory_info')
                    if mem:
                        total_rss += mem.rss
                        total_vms += mem.vms
                    
                    # 获取 IO 和联合主键
                    io = p.info.get('io_counters')
                    create_time = p.info.get('create_time', 0)
                    if io:
                        proc_key = (p.pid, create_time)
                        current_data[proc_key] = {
                            'bytes': getattr(io, 'read_bytes', 0),
                            'chars': getattr(io, 'read_chars', 0)
                        }
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
        return current_data, total_rss, total_vms

    def _record_baseline(self):
        """记录监控启动时的初始 I/O 状态"""
        current_data, _, _ = self._get_process_io_data()
        self.baseline_io = current_data
        
        # 初始状态下，最高水位线等于基线
        for key, io_data in current_data.items():
            self.max_seen_io[key] = io_data.copy()

    def get_metrics(self):
        """
        获取从监控启动以来的真实 I/O 增量和当前内存
        """
        current_data, total_rss, total_vms = self._get_process_io_data()
        
        # 1. 更新观察到的最高 I/O 水位线
        for key, io_data in current_data.items():
            if key not in self.max_seen_io:
                # 这是一个在监控启动后才“新出生”的进程
                self.max_seen_io[key] = {'bytes': 0, 'chars': 0}
            
            # 更新最高记录，防止瞬时系统波动
            self.max_seen_io[key]['bytes'] = max(self.max_seen_io[key]['bytes'], io_data['bytes'])
            self.max_seen_io[key]['chars'] = max(self.max_seen_io[key]['chars'], io_data['chars'])

        # 2. 计算从监控启动以来的真实增量
        incremental_bytes = 0
        incremental_chars = 0
        
        for key, max_io in self.max_seen_io.items():
            # 获取该进程在监控启动时的初始值（如果是新进程，初始值为 0）
            baseline = self.baseline_io.get(key, {'bytes': 0, 'chars': 0})
            
            incremental_bytes += (max_io['bytes'] - baseline['bytes'])
            incremental_chars += (max_io['chars'] - baseline['chars'])

        return {
            "rss_mb": total_rss / (1024 * 1024),
            "vms_mb": total_vms / (1024 * 1024),
            "read_bytes_inc_mb": incremental_bytes / (1024 * 1024),
            "read_chars_inc_mb": incremental_chars / (1024 * 1024)
        }


def _monitor_worker(target_pid, interval=1.0):
    """
    独立监控进程：实时汇总当前存活的主进程及所有子进程的 内存 和 I/O
    """
    # 字典：专门用来记录每一个出现过的 PID 的“历史最高 I/O 水位线”
    pid_max_read = {}
    pid_max_read_chars = {}
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    def _kill_master():
        try:
            os.kill(target_pid, signal.SIGKILL)
        except OSError:
            pass
        sys.exit(1)

    uid_monitor = UIDResourceMonitor()
    sample_round = 0
    while True:
        try:
            parent = psutil.Process(target_pid)
            processes = [parent] + parent.children(recursive=True)
            
            # 每轮循环重置内存计数器，因为内存是实时的
            total_rss = 0
            total_vms = 0
            total_swap = 0
            active_count = 0

            # 1. 遍历当前时刻存活的进程
            for p in processes:
                try:
                    # 【内存统计】只算当前还活着的
                    mem = p.memory_full_info()
                    total_rss += mem.rss
                    total_vms += mem.vms
                    total_swap += getattr(mem, 'swap', 0)
                    
                    # 【I/O 统计】更新该进程的历史最高读取量
                    # 使用 max() 防止极个别情况下系统计数器跳变
                    io = p.io_counters()
                    pid_max_read[p.pid] = max(pid_max_read.get(p.pid, 0), io.read_bytes)
                    pid_max_read_chars[p.pid] = max(pid_max_read_chars.get(p.pid, 0), io.read_chars)
                    
                    active_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # 瞬间死掉的进程被忽略，它的 I/O 已经在上一秒被记录过了
                    continue

            # 2. 计算累计 I/O（把字典里存的活着的、死掉的所有进程的最高 I/O 加起来）
            total_cumulative_read = sum(pid_max_read.values())
            total_cumulative_write = sum(pid_max_read_chars.values())

            # 3. 单位换算为 MB
            rss_mb = total_rss / (1024 * 1024)
            vms_mb = total_vms / (1024 * 1024)
            swap_mb = total_swap / (1024 * 1024)
            read_mb = total_cumulative_read / (1024 * 1024)
            write_mb = total_cumulative_write / (1024 * 1024)
            
            try:
                logger.info(f"{active_count}, {total_rss}, {rss_mb:.2f}, {vms_mb + swap_mb}, {total_vms}, {vms_mb:.2f}, {total_swap}, {swap_mb:.2f}, " \
                            f"{total_cumulative_read}, {read_mb:.2f}, {total_cumulative_write}, {write_mb:.2f}")
            except:
                msg = f'Failed monitor the main process which is MUST, exiting ...'
                logger.error(msg)
                _kill_master()

        except psutil.NoSuchProcess:
            break
        except Exception as e:
            print('Some unexpected exception: ', e)
            _kill_master()
        sample_round += 1
        if sample_round % 5 == 0:
            try:
                r = uid_monitor.get_metrics()
                logger.info(f"uid metric: {r['rss_mb']}, {r['vms_mb']}, {r['read_bytes_inc_mb']}, {r['read_chars_inc_mb']}")
            except:
                msg = f'Failed monitor the main process which is MUST, exiting ...'
                print(msg)
                logger.error(msg)
                _kill_master()
        time.sleep(interval)


def start_monitor():
    log_dir = "audit-logs"
    interval = 2
    os.makedirs(log_dir, exist_ok=True)
    
    main_process = psutil.Process()
    time_gap = time.time() - main_process.create_time()
    if time_gap > 2:
        raise Exception(f'Auditor monitor have not start with the main process, time gap: {time_gap}')
    for script in main_process.cmdline():
        pwd = main_process.cwd()
        script_path = os.path.join(pwd, script)
        if os.path.exists(script_path) and script.endswith('.py'):
            break

    entrypoint_content = Path(script_path).read_text()
    lines = entrypoint_content.splitlines()

    required_code = ['#!/usr/bin/env python3', 'import audit_monitor']
    code_valid = False
    for idx, c in enumerate(required_code):
        code_valid = c == lines[idx]
        # code_valid = True
        if not code_valid:
            tips = f"ERROR: The first two lines of the entry program ({script}) must start with: \n\n" \
                    f"#!/usr/bin/env python3\n import audit_monitor \n\n" \
                    f"------------------------------\n\n" \
                    f"This is yours:\n\n"
            tips += lines[0] + '\n'
            tips += lines[1] + '\n'
            print(tips)
            exit(0)

    monitor_p = multiprocessing.Process(
        target=_monitor_worker, 
        args=(main_process.pid, interval)
    )
    monitor_p.daemon = True 
    monitor_p.start()
    print(f"Audit monitor started ...",  time_gap, main_process.pid, main_process.create_time(), monitor_p.pid, time.time())

    def on_main_exit():
        print("[Main] process is exiting ...")
        monitor_p.join(timeout=1)

        file_loading_logger = secure_logger.get_logger('file-loading')
        training_logger = secure_logger.get_logger('training')
        logger.archive_gracefully()
        file_loading_logger.archive_gracefully()
        training_logger.archive_gracefully()
    atexit.register(on_main_exit)

start_monitor()
