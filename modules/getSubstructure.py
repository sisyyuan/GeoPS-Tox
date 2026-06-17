import argparse
from chemistryProcess import get_substructure


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Function Aux')
    parser.add_argument('--smile',default='O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1', type = str)
    parser.add_argument(
        '--method', choices = ['brics', 'recap'], default = 'brics'
    )
    args = parser.parse_args()

    tx = get_substructure(smile = args.smile, decomp = args.method)
    print(f'{args.smile}\t{str(tx)}')