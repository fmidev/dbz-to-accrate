import argparse
import json

import observation_dbzh_to_accr
import forecast_dbzh_to_accr


def main():

    config_file = f"config/{options.config}.json"
    with open(config_file, "r") as jsonfile:
        config = json.load(jsonfile)
        filetype = config["input"]["filetype"]
      
    if filetype == "forecast":
        forecast_dbzh_to_accr.run(options.timestamp, options.config)

    elif filetype == "observation":
        observation_dbzh_to_accr.run(options.timestamp, options.config)
    
                
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
    
    options = parser.parse_args()
    args = vars(options)
    main()
