from Slurm import *    # NOTE: Make sure to import the gene too!
import logging
import argparse
import traceback
import sys


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-id", type=int, required=True, action="store")
  args = parser.parse_args()
  id = args.id

  # Create a logging object
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.ERROR)

  # Create a file handler
  handler = logging.FileHandler('error.log')

  # Set the logging format
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)

  # Add the file handler to the logger
  logger.addHandler(handler)

  try:
    model = TestModel("params.yaml")
    client = SlurmClient(model, sim_dir="test_sim", gene_id=args.id, server_callback="server_callback.sh")
    client.run(verbose=True)
  except Exception as e:
    logger.error("An error occurred: %s", traceback.format_exc())
