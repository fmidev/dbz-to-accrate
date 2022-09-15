import argparse

import composite_dbzh_to_accr
import fmippn_dbzh_to_accr


def main():

    if options.filetype == "forecast":
        fmippn_dbzh_to_accr(**args)

    elif options.filetype == "composite":
        composite_dbzh_to_accr(**args)
    
                
if __name__ == '__main__':
    #Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp',
                        type = str,
                        default = '202201170700',
                        help = 'Input timestamp')
    parser.add_argument('--config',
                        type = str,
                        default = 'ravake_composite',
                        help = 'Config file to use.')
    parser.add_argument('--filetype',
                        type = str,
                        default = 'composite',
                        help = '[forecast|composite]')

    
    options = parser.parse_args()
    args = vars(options)
    main()
