from setuptools import setup, find_packages

def main():
    setup(
        name = 'rpcf_post',
        packages = find_packages(include = ['couette', 'couetteModel', 'rpcf']),
        install_requires = ['numpy',
                            'scipy',
                            'matplotlib',
                            'h5py']
    )

if __name__ == '__main__':
    main()