from data_provider.data_loader import (
    Dataset_Covid,
    Dataset_Custom,
    Dataset_Pred,
    Dataset_Custom_,
    Dataset_HDF,
)
from torch.utils.data import DataLoader

data_dict = {
    "ETTh1": Dataset_Custom_,  # Dataset_ETT_hour,
    "ETTm1": Dataset_Custom_,
    "traffic": Dataset_Custom,
    "electricity": Dataset_Custom_,
    "exchange": Dataset_Custom_,
    "weather": Dataset_Custom_,
    "covid": Dataset_Covid,
    "ECG": Dataset_Custom_,
    "metr": Dataset_Custom_,
}


def data_provider(args, flag):
    if args.data in data_dict.keys():
        Data = data_dict[args.data]
    else:
        Data = Dataset_HDF
    print(Data)
    timeenc = 0 if args.embed != "timeF" else 1
    train_only = args.train_only

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if isinstance(Data, Dataset_HDF.__class__):
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            train_only=train_only,
            trunc=2000,
        )
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
            train_only=train_only,
        )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
