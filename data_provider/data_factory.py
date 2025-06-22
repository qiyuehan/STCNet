
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_minute, Dataset_ETT_hour

data_dict = {
    'ETTh': Dataset_ETT_hour,
    'ETTm': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}



def data_provider(args, flag):
    Data = data_dict[args.data_class]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # 10  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    # print("=====================", args.data_class)
    if 'ETT' in args.data_class:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last)
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            # seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last)

    return data_set, data_loader