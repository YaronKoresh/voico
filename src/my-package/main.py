import argparse 
import logging 
import sys 
import os 
from pathlib import Path 

sys .path .append (str (Path (__file__ ).parent .parent ))

def setup_logging (verbose :bool ):
    level =logging .DEBUG if verbose else logging .INFO 
    logging .basicConfig (
    level =level ,
    format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt ='%H:%M:%S'
    )

def parse_args ():
    parser =argparse .ArgumentParser (description ="My Package: Example Package")

    parser .add_argument ("input_file",help ="Path to input file")
    parser .add_argument ("-t","--target",help ="Path to target file ",default =None )
    parser .add_argument ("-o","--output",help ="Path to output file",default =None )

    parser .add_argument ("-v","--verbose",action ="store_true",help ="Enable debug logging")

    return parser .parse_args ()

def main ():
    args =parse_args ()
    setup_logging (args .verbose )

    if not os .path .exists (args .input_file ):
        print (f"Error: Input file '{args.input_file}' not found.")
        sys .exit (1 )

    if args .target and not os .path .exists (args .target ):
        print (f"Error: Target file '{args.target}' not found.")
        sys .exit (1 )

    if args .output :
        out_path =args .output 
    else :
        base ,ext =os .path .splitext (args .input_file )
        if args .target :
            target_name =Path (args .target ).stem 
            out_path =f"{base}_to_{target_name}{ext}"
        else :
            print (f"Error: Please specify a target/output path.")
            sys .exit (1 )

    try :
        print ("OK")

    except Exception as e :
        logging .error (f"Error: {e}",exc_info =args .verbose )
        sys .exit (1 )

if __name__ =="__main__":
    main ()