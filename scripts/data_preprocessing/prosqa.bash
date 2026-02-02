mkdir -p ./datasets/prosqa

wget -O ./datasets/prosqa/train.json \
https://raw.githubusercontent.com/facebookresearch/coconut/main/data/prosqa_train.json

wget -O ./datasets/prosqa/valid.json \
https://raw.githubusercontent.com/facebookresearch/coconut/main/data/prosqa_valid.json

wget -O ./datasets/prosqa/test.json \
https://raw.githubusercontent.com/facebookresearch/coconut/main/data/prosqa_test.json
