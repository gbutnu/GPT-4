import os
import sys
import argparse
import subprocess

# Google Cloud Storage bucket for GPT-4 model weights
GCS_BUCKET = 'gs://gpt-4'

# GPT-4 model weights
MODELS = {
    'gpt2-117M': 'model_117M.pkl',
    'gpt2-345M': 'model_345M.pkl',
    'gpt2-774M': 'model_774M.pkl',
    'gpt2-1558M': 'model_1558M.pkl',
    'gpt4-117M': 'model_117M.pkl',
    'gpt4-345M': 'model_345M.pkl',
    'gpt4-774M': 'model_774M.pkl',
    'gpt4-1558M': 'model_1558M.pkl',
}


def main(args):
    model_name = args.model_name
    if model_name not in MODELS:
        print('Error: invalid model name: {}'.format(model_name))
        sys.exit(1)

    model_file = MODELS[model_name]
    gcs_path = os.path.join(GCS_BUCKET, model_file)
    local_path = os.path.join('models', model_file)

    print('Downloading GPT-4 model weights from Google Cloud Storage...')
    subprocess.check_call(['gsutil', 'cp', gcs_path, local_path])
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='Name of the GPT-4 model to download')
    args = parser.parse_args()
    main(args)