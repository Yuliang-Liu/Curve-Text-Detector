it=$1

./tools/test_net.py --gpu 1 \
  --def models/ctd/test_ctd_tloc.prototxt \
  --net output/ctd_tloc.caffemodel \
  --imdb ctw1500_test \
  --cfg experiments/cfgs/rfcn_ctd.yml \
  --test_label data/ctw1500/test/test_label_curve.txt \
  --test_image data/ctw1500/test/test.txt \
  # --vis 
