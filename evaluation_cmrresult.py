import os
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

from src import utils

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', required=True, help='Dump dir to save model checkpoint [default: log]')
FLAGS = parser.parse_args()

LOGDIR = FLAGS.log_dir


class Evaluator(object):
    def __init__(self):

        save_path = Path(LOGDIR) / "info_test"

        pred_path = "prediction.txt"
        print("[INFO] load predictions results")
        with open(str(save_path / pred_path), "r") as f:
            self.lines = f.readlines()
        sections, last = self.calculate_sections(self.lines)
        print("[INFO] load finished")
        self.sections = sections
        print(sections)


    def evaluation_seeds(self):
        tss = []
        rss = []
        count = 0
        for tag, (num, start, pose_t) in self.sections.items():
            start += 1
            init_start = start
            ts = []
            rs = []
            count = 0
            for vis_t in tqdm(range(num)):
                start = init_start + pose_t * vis_t
                # init_extrinsic = np.array(self.lines[start + 1].strip('\n').split(' '), np.float32).reshape(3, 4)
                pred_extrinsic = np.array(self.lines[start + pose_t - 2].strip('\n').split(' '),
                                          np.float32).reshape(3, 4)
                gt_extrinsic = np.array(self.lines[start + pose_t - 1].strip('\n').split(' '),
                                        np.float32).reshape(3, 4)

                initial_error = utils.mult_extrinsic(pred_extrinsic, utils.inv_extrinsic(gt_extrinsic))

                t_diff = np.linalg.norm(initial_error[:3, 3], 2, -1)

                r_diff = initial_error[:3, :3]
                angles_diff = np.arccos(np.clip((np.trace(r_diff) - 1) / 2,-1,1)) * 180. / np.pi
                if count > 4540:
                    break
                rs.append(angles_diff)
                ts.append(t_diff)
                count += 1

            print(f"RRE:{np.array(rs).mean()}+-{np.array(rs).std()} RTE:{np.array(ts).mean()}+-{np.array(ts).std()}")
            print(f"RRE median:{np.median(np.array(rs))} RTE median:{np.median(np.array(ts))}")
            
            tss.append(np.array(ts,np.float32))
            rss.append(np.array(rs,np.float32))

            count += 1
            if count == 10:
                break
            # if len(ts) == 4541:
            #     tss.append(np.array(ts,np.float32))
            #     rss.append(np.array(rs,np.float32))

        rss = np.stack(rss)
        tss = np.stack(tss)
        print(f"RRE:{rss.mean():.3f}+-{rss.std():.3f} RTE:{tss.mean():.3f}+-{tss.std():.3f}")
        print(f"RE median: {np.median(rss, axis=1).mean():.3f} TE median: {np.median(tss, axis=1).mean():.3f}")
        np.save('kd_rot', np.array(rss))
        np.save('kd_trans', np.array(tss))

    def calculate_sections(self, lines):
        count = -1
        section = {}
        count2 = 0
        last = None
        while count + len(lines) >= 0:
            # if abs(count) == len(self.lines):
            #     break
            if "section" in lines[count]:
                #if count2 % 4 == 0:  # no coarse:
                    #section[lines[count].strip("[section sign] prediction on ")[:19]] = (count2 // 4, count, 4)
                if count2 % 5 == 0:  # coarse
                    section[lines[count].strip("[section sign] prediction on ")[:19]] = (count2 // 5, count, 5)
                else:
                    continue
                count2 = 0
                if last is None:
                    last = lines[count].strip("[section sign] prediction on ")[:19]

            else:
                count2 += 1
            count -= 1
        # print(section)
        return section, last


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.evaluation_seeds()