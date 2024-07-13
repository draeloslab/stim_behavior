import argparse
import sys

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--project_root', type=str, required=True,
      help='Root directory of the project. (eg: "/home/<username>/Code/stim_behavior/")')
    parser.add_argument( '--input_dir', type=str, required=True,
      help='Directory of the downloaded svd data of mice. (eg: "/home/<username>/Data/raw/mouse-cshl")')
    parser.add_argument( '--output_dir', type=str, required=True,
      help='Directory to store the processed videos. (eg: "/home/<username>/Data/processed/mouse-cshl")')
    return parser

def main(args):
    sys.path.append(args.project_root)
    from models.mice_data_loader import MiceDataLoader
    input_dir = args.input_dir
    output_dir = args.output_dir

    ###
    mouse_id = 'mSM61'
    dates = ['05-Sep-2018', '06-Jun-2018', '07-Sep-2018', '08-Jun-2018', '11-Sep-2018', '12-Jun-2018']
    cams = ['1', '2']

    for date in dates:
        for cam in cams:
            data_ld = MiceDataLoader(input_dir, output_dir, verbose=False)
            data_ld.init_file(mouse_id, date, cam)
            data_ld.merge_svd(end_V=1000)
            data_ld.save_video()

if __name__ == "__main__":    
    main(create_parser().parse_args())