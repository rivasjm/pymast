from bs4 import BeautifulSoup


def parse_results(file):
    wcrts = {}
    with open(file, "r") as f:
        soup = BeautifulSoup(f, features="xml")
        for trans in soup.findAll("mast_res:Transaction"):
            name = (trans["Name"])
            wcrt = max([float(x["Time_Value"]) for x in trans.findAll("mast_res:Global_Response_Time")])
            wcrts[name] = wcrt
    return wcrts


if __name__ == '__main__':
    print(parse_results("test-out.xml"))
