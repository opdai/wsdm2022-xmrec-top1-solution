# set_jupyter.py

from notebook.auth import passwd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='set-passwd')
    parser.add_argument('--passwd', default='wsdm666', type=str)
    parser.add_argument('--port', default='8010', type=str)
    parser.add_argument('--jpt_base_dir', default='./', type=str)
    args = parser.parse_args()
    # print(F"# Your PassWord: {args.passwd}")
    # print(passwd(args.passwd))
    tgt_str = f"""c.NotebookApp.ip='*' # or '0.0.0.0'\nc.NotebookApp.notebook_dir = u'{args.jpt_base_dir}'\nc.NotebookApp.open_browser = False\nc.NotebookApp.password = '{passwd(args.passwd)}'\nc.NotebookApp.port = {args.port}\nc.NotebookApp.allow_root = True"""
    print(tgt_str)