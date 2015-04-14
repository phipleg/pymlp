from sys import stdout

def printi(str):
    stdout.write("\r" + str)
    stdout.flush()

def to_ht(seconds):
    if seconds <= 60:
        return "{:2.0f}s".format(seconds)
    if seconds <= 60*60:
        return "{:2.1f}m".format(seconds / 60.0)
    if seconds <= 60*60*24:
        return "{:2.1f}h".format(seconds / (60.0 * 60))
    return "{:.1f}d".format(seconds / (60.0 * 60 * 24))

def bar(x):
    width = 20
    xi = int(x*width)
    s = "["
    for i in xrange(xi):
        s += '='
    if xi < width:
        s += ">"
    for i in xrange(xi+1, width):
        s += '-'
    s += "]"
    return s
   
