

def analyze_log(file):
    with open(file) as f:
        for line in f.readlines():
            if has_anomaly(line):
                print(line.rstrip("\n"))


def has_anomaly(line):
    tools = line.split("->")[1].strip()
    if "bf" not in tools and tools:
        return True
    return False


if __name__ == '__main__':
    analyze_log("243-bf-hol-2_log.txt")
