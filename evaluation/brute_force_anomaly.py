

def analyze_log(file):
    with open(file) as f:
        for line in f.readlines():
            tools = line.split("->")[1].strip()
            if "bf" not in tools and tools:
                print(line.rstrip("\n"))


if __name__ == '__main__':
    analyze_log("243-bf-hol-2_log.txt")
