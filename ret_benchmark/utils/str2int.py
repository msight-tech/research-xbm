def str2int(s):
    if s.isdigit() or s.startswith("-"):
        return int(s)
    else:
        ascii_str = "".join([str(ord(k))[0] for k in s])
        return -int(ascii_str)
