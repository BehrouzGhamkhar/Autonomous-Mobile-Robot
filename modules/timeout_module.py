import signal

def timeout_handler(signum, frame):
    raise Exception("Timed out!")


def run_with_timeout(timeout, func, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = func(*args, **kwargs)
    except Exception:
        print(f"{func.__name__} execution exceeded {timeout} seconds!")
        result = None 
    finally:
        # Cancel the alarm
        signal.alarm(0)
    
    return result
