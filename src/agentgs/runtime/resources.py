import os
import signal

import psutil


def tree_rss(pid):
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for proc in processes:
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def kill_tree(pid):
    try:
        children = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if os.name != "nt":
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            processes = [*psutil.Process(pid).children(recursive=True), psutil.Process(pid)]
        except psutil.NoSuchProcess:
            return
        for proc in reversed(processes):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
