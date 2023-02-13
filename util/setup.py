from setuptools import setup, find_packages

def main():
    setup(
        name = 'rpcf_util',
        packages = find_packages(include = ['rpcf_util']),
        install_requires = ['numpy',
                            'matplotlib']
    )

if __name__ == '__main__':
    main()