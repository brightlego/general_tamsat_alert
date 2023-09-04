def progress_bar(iterator, msg, value_to_str=lambda i, imax, val: "", width=32, in_progress_bar=False):
    if in_progress_bar:
        yield from enumerate(iterator)
        return
    prev_str = ""
    print(msg, end=":\t")
    for i, val in enumerate(iterator):
        proportion = i/len(iterator)
        bar_count = int(proportion*(width-1))

        sub_bar = int(proportion*(width-1)*8) % 8
        if proportion >= 0.99999999:
            sub_bar = 8
        bar = "[" + "█"*bar_count + " ▏▎▍▌▋▊▉██"[sub_bar] + " "*(width-bar_count-1) + "]"

        prev_str_len = len(prev_str)
        prev_str = (f"{i+1:>{len(str(len(iterator)))}}/{len(iterator)} {bar} {proportion*100:.1f}% "
                    f"{value_to_str(i, len(iterator), val)}")
        print("\b"*prev_str_len, end="")
        print(prev_str, end="")
        yield i, val
    print("\b"*len(prev_str), end="")
    print(f"{len(iterator)}/{len(iterator)} [{'█'*width}] 100%")
