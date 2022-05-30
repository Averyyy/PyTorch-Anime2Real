import os

if os.path.exists('A'):
    for f in os.listdir('A'):
        if 'real' in f:
            os.remove(os.path.join('A', f))
if os.path.exists('B'):
    for f in os.listdir('B'):
        if 'real' in f:
            os.remove(os.path.join('B', f))