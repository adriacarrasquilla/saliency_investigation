python demo.py --image $1 > /dev/null 2>&1
cd ../generative-inpainting-pytorch

python test_tf_model.py --image ../saliency_investigation/demo_outputs/org.png \
    --mask ../saliency_investigation/demo_outputs/mask_in.png \
    --output ../saliency_investigation/demo_outputs/infilled_in.png \
    --model-path torch_model.p

python test_tf_model.py --image ../saliency_investigation/demo_outputs/org.png \
    --mask ../saliency_investigation/demo_outputs/mask_out.png \
    --output ../saliency_investigation/demo_outputs/infilled.png \
    --model-path torch_model.p

cd ../saliency_investigation
python plot_output.py
python compare_predictions.py > /dev/null 2>&1
