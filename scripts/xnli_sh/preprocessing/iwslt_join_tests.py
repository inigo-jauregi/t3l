import argparse


def main(args):

    src_list = []
    tgt_list = []
    for subdataset in args.subdatasets:
        subdataset = "".join(subdataset)
        src_docs = open(args.dataset_path + '/' + subdataset + '.' + args.src)
        for line in src_docs:
            src_list.append(line.strip())
        src_docs.close()
        tgt_docs = open(args.dataset_path + '/' + subdataset + '.' + args.tgt)
        for line in tgt_docs:
            tgt_list.append(line.strip())
        tgt_docs.close()

    print(len(src_list))
    print(len(tgt_list))
    assert len(src_list) == len(tgt_list), "Different number of sentences!"

    joined_src_write = open(args.dataset_path + '/' + args.outname + '.' + args.src, 'w')
    joined_tgt_write = open(args.dataset_path + '/' + args.outname + '.' + args.tgt, 'w')

    for i in range(len(src_list)):
        joined_src_write.write(src_list[i])
        joined_tgt_write.write(tgt_list[i])
        if i < len(src_list) - 1:
            joined_src_write.write('\n')
            joined_tgt_write.write('\n')

    joined_src_write.close()
    joined_tgt_write.close()


my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('--dataset_path', type=str, required=True,
                       help="Dataset path.")
my_parser.add_argument('--subdatasets', nargs="+", type=list, required=True,
                       help="Year of the dataset.")
my_parser.add_argument('--outname', type=str, required=True,
                       help="Output joined dataset name.")
my_parser.add_argument('--src', type=str, required=True,
                       help="Source language prefix.")
my_parser.add_argument('--tgt', type=str, required=True,
                       help="Target language prefix.")

if __name__ == "__main__":
    my_args = my_parser.parse_args()
    main(my_args)
