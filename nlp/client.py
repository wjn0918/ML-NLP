
import hprose


def main():
    client = hprose.HttpClient('http://127.0.0.1:8181/')
    with open('lib/cs.txt', encoding='utf8') as rd:
        doc = rd.read()
    print(doc)
    print(client.pre_txt(doc))


if __name__ == '__main__':
    main()
