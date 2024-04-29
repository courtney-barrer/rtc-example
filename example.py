from time import sleep
import rtc
import numpy as np
from datetime import timedelta

def print_n_last_lines(s: str, n: int = 10):
    lines = s.split('\n')
    for l in lines[-n:]:
        print(l)

# create 2 slope offsets buffer.
slope_offsets = np.ones((2, 15), dtype=np.float32)
# set the first slope offset to 1 and the second to 2
slope_offsets[1] = 2
slope_offsets[0] = 1

r = rtc.RTC()

# init the rtc. Could have been done using constructor but requires to code it.
r.set_slope_offsets(slope_offsets[0])
r.set_gain(1.1)
r.set_offset(2)

# none of the above commands are executed yet until we commit.
# It's safe to do it because the rtc is not running yet.
r.commit()

# Create an async runner. This component will run the rtc in a separate thread.
runner = rtc.AsyncRunner(r, period = timedelta(microseconds=1000))


runner.start()

sleep(1)
print_n_last_lines(runner.flush(), 6)


r.set_slope_offsets(slope_offsets[1])
r.set_gain(0)
r.set_offset(-1)

# request a commit. The runner will commit the new values at the next iteration.
r.request_commit()

sleep(.2)

# pause keep the thread alive but stop the execution of the rtc.
# this can be resume later using runner.resume()
runner.pause()

# get the output of the runner but just keep the last 6 lines.
print_n_last_lines(runner.flush(), 6)

# kill the thread. A new thread can still be recreated using `start` later.
runner.stop()

# `del runner`` will also stop the thread.
