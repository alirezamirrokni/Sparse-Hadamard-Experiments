import fcntl
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


def process_identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


@contextmanager
def timing_slot(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / "timing-queue.sqlite", timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS queue "
        "(ticket INTEGER PRIMARY KEY AUTOINCREMENT, pid INTEGER, identity TEXT)"
    )
    with connection:
        cursor = connection.execute(
            "INSERT INTO queue(pid, identity) VALUES (?, ?)",
            (os.getpid(), process_identity(os.getpid())),
        )
        ticket = cursor.lastrowid
    lock = (directory / "timing.lock").open("a")
    acquired = False
    try:
        while not acquired:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                stale = [
                    (number,)
                    for number, pid, identity in connection.execute(
                        "SELECT ticket, pid, identity FROM queue"
                    )
                    if process_identity(pid) != identity
                ]
                connection.executemany("DELETE FROM queue WHERE ticket=?", stale)
                first = connection.execute("SELECT MIN(ticket) FROM queue").fetchone()[
                    0
                ]
                if first == ticket:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                    except BlockingIOError:
                        pass
            if not acquired:
                time.sleep(0.1)
        yield
    finally:
        if acquired:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        with connection:
            connection.execute("DELETE FROM queue WHERE ticket=?", (ticket,))
        connection.close()
