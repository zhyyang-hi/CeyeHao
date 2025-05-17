from utils.data_process import img2fp_coords
from utils.io import mkdirs
import cv2

from config.config import parse_args
from tools.ga import GASearch


if __name__ == "__main__":
    args = parse_args()
    TGT_PTH = args.tgt_pth

    tgt_img = cv2.imread(TGT_PTH)
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)
    tgt_img = cv2.resize(tgt_img, (200, 200))
    _, tgt_img = cv2.threshold(tgt_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tgt_img = img2fp_coords(tgt_img)



    searcher = GASearch(
        population=args.population,
        elitism_ratio=args.elitism,
        mutation=args.mutation,
        prob_select=args.prob_select,
        metric=args.metric,
    )
    searcher.set_task(
        tgt_fp_tensor=tgt_img,
        min_num_hao=args.min_num_hao,
        max_num_hao=args.max_num_hao,
        restrict_flow_ratio=args.restrict_flow_ratio,
        log_dir=mkdirs(args.log_dir, remove_old=True),
    )
    searcher.run_task()
