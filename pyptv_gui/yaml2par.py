import yaml


class YAML:
    def __init__(self):
        filename = '/Users/alex/Documents/OpenPTV/test_cavity/parameters/ptv.yaml'
        with open(filename) as f:
            yaml_args = yaml.load(f)


        for k,v in yaml_args.items():
            if isinstance(v,list) and len(v) > 1: # multi line
                setattr(self,k,[])
                tmp = []
                for i,item in enumerate(v):
                    tmp.append(item)
                setattr(self, k, tmp)

            setattr(self, k, v)


if __name__ == '__main__':
    yaml_test = YAML()
    tmp = yaml_test.__dict__
    for k,v in tmp.items():
        if k is not 'path':
            print(k,v)

