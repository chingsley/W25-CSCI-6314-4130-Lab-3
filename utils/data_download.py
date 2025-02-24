import requests
import gzip
from io import BytesIO

def download_data_file_from_seart(file_type, **kwargs)->bool:
    '''Download data file from SEART'''

    url = f'https://seart-ghs.si.usi.ch/api/r/download/{file_type}?'
    for k, v in kwargs.items():
        if url[-1] != '?':
            url += '&'
        url += str(k) + '=' + str(v)
    r = requests.get(url, verify=False, timeout=60)

    if r.status_code != 200:
        print(f'Error {r.status_code} when downloading data file from SEART')
        return False
    
    with open(f'./data/new_result.{file_type}', 'wb') as f:
        compressed_content = r.content
        with gzip.GzipFile(fileobj=BytesIO(compressed_content), mode='rb') as gz_file:
            decompressed_content = gz_file.read()

        f.write(decompressed_content)
    return True

if __name__=='__main__':
    download_data_file_from_seart('json', nameEquals=False,language="Java",commitsMin=5000,contributorsMin=2,issuesMin=1,pullsMin=1,
                                  committedMin="2023-12-24",committedMax="2024-01-21",starsMin=2000,forksMin=10)