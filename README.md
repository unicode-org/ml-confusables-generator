# Confusable Detector
The wide range of characters supported by unicode poses security vulnerabilities and allows spoofing attack (IDN homograph attack). The security mechanism listed in UTS#39 (UTS #39) uses confusable data (https://www.unicode.org/Public/security/latest/confusables.txt) to combat such attacks. A pair of confusables is a pair of characters which might be used in a spoofing attack due to their similarity (for example ‘ν’ and ‘v’). The purpose of this project is to identify novel pairs of confusables by computing a custom distance metric on a large scale.


## Set up
- `git submodule update --init --recursive`: make sure all submodules are updated.
- `./setup.sh`: install all python packages.

