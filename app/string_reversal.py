def reverse_string_exclude_special(s):
    letters = [c for c in s if c.isalnum()]
    reversed_s = []
    for c in s:
        if c.isalnum():
            reversed_s.append(letters.pop())
        else:
            reversed_s.append(c)
    return ''.join(reversed_s)