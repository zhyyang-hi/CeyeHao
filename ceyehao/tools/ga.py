# multi-process ga.
# open a log and two predict throughout the search. in each gen spawn several processes to run the process.
# first gen, sample pin and microchannel.
# bacth calc fitness. using IoU? collect fitness and rank them
# new gen top 5 = elism
# slection: stochastic tournament the rest 95 based on ranked fitness
# mutation:
# uniform random (num_mutation) index for mutation.
# chromosome shape: (num_chr, coord + num_pos + pin) float 32
# fitness shape: (num_chr, score)
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
from easydict import EasyDict
from queue import Empty
import warnings
from tools.acc_metrics import ssim

from ceyehao.utils.io import dump_cfg_yml
from ceyehao.tools.infer import TTPredictor
from ceyehao.tools.acc_metrics import PixelwiseAccuracy
from ceyehao.utils.utils import mkdirs, get_logger, Timer
from ceyehao.utils.data_generate import (
    HAOParamSampler as HPS,
    laplace_sampler,
    laplace_quantile,
)
from ceyehao.utils.data_process import (
    p_transform,
    gen_pin_tensor,
    InflowCalculator,
    tt_synth,
)
from ceyehao.utils.visualization import (
    create_profile_figure,
    plot_fp_tensor,
)


class GASearch:
    def __init__(
        self,
        population=100,
        elitism_ratio=0.03,
        mutation=0.35,
        metric: str = "IoU",
        prob_select: bool = False,
    ):
        mp.set_start_method("spawn")
        self.rng = np.random.default_rng(1234)
        self.hcp_generator = HPS()  # HAO coordinate param sampler
        self.pin_param_calc = InflowCalculator(verbose=False)
        self.cfg = {}
        self.cfg["fp_res"] = (200, 200)
        self.cfg["population"] = population
        self.cfg["lap_miu"] = 0  # miu of laplace distribution used in crossover
        self.cfg["lap_b"] = 0.35
        self.cfg["chr_mutation_rate"] = mutation
        self.cfg["gene_mutation_rate"] = mutation
        self.cfg["gaussian_mut_scale"] = 0.5
        self.cfg["elitism_ratio"] = elitism_ratio
        self.cfg["tournament_size"] = 4
        self.cfg["max_generation"] = 100
        self.cfg["stall_limit"] = 999  # no limit for now. lets run.
        self.cfg["num_pred_worker"] = 3  # for parallel computing
        self.cfg["num_eval_worker"] = 3
        self.cfg["hao_meta"] = self.hcp_generator.coord_meta
        self.cfg["prob_select"] = prob_select
        self.cfg["metric"] = metric

        self.set_metric(metric)
        # here define some variables that are repeatedly used in the computations
        self.num_elite = round(self.cfg["elitism_ratio"] * self.cfg["population"])
        self.num_offspring = self.cfg["population"] - self.num_elite
        self.num_mut_chr = round(self.num_offspring * self.cfg["chr_mutation_rate"])
        self.xi0_half_interval = 2 * self.hcp_generator.coord_meta["xi0_half_interval"]
        num_seg = self.hcp_generator.coord_meta["num_seg"]
        yi_a = self.hcp_generator.coord_meta["yi_a"]
        yi_b = self.hcp_generator.coord_meta["yi_b"]
        dx_var = self.hcp_generator.coord_meta["var_delta_x"]
        dx_mean = self.hcp_generator.coord_meta["mean_delta_x"]
        self.sw_weight_min = yi_b  # sw is segment width
        self.sw_weight_max = yi_a - num_seg + self.sw_weight_min
        self.hao_sw_max = yi_a / (
            self.sw_weight_max + (num_seg - 1) * self.sw_weight_min
        )
        self.hao_sw_min = yi_b / (
            (num_seg - 1) * self.sw_weight_max + self.sw_weight_min
        )
        self.hao_sw_interval = self.hao_sw_max - self.hao_sw_min
        self.dx_min, self.dx_max = dx_mean - dx_var, dx_mean + dx_var
        self.lap_xo_beta_max = laplace_quantile(
            0.9975, self.cfg["lap_miu"], self.cfg["lap_b"]
        )
        self.lap_xo_beta_min = laplace_quantile(
            0.0025, self.cfg["lap_miu"], self.cfg["lap_b"]
        )

    def set_task(
        self,
        tgt_fp_tensor: np.ndarray,
        min_num_hao,
        max_num_hao,
        log_dir,
        restrict_flow_ratio,
    ):
        """
        pf is single channel, binary image of shape (h, w), uint8"""
        assert (
            tgt_fp_tensor.shape[0:2] == self.cfg["fp_res"]
        ), f"the target image size is{tgt_fp_tensor.shape[0:2]}, but the expected size is {self.cfg['fp_res']}"
        self.log_dir = log_dir
        self.cfg["min_num_hao"] = min_num_hao
        self.cfg["max_num_hao"] = max_num_hao
        self.cfg["restrict_flow_ratio"] = restrict_flow_ratio
        if restrict_flow_ratio:
            target_flow_ratio = InflowCalculator.estimate_fr_ratio(tgt_fp_tensor)
            self.fr_range = (target_flow_ratio * np.array([0.7, 1.3])).clip(0.1, 0.8)
        else:
            self.fr_range = np.array((0.1, 0.8))
        self.tgt_fp = tgt_fp_tensor[..., np.newaxis]  # shape (H, W, C)
        self.tgt_fp = np.array(
            [self.tgt_fp] * (self.cfg["max_num_hao"] - self.cfg["min_num_hao"] + 1)
        )

    def set_metric(self, metric):
        if metric == "IoU":
            self.metric = PixelwiseAccuracy(EasyDict({"op_flags": ["IoU"]}))
        elif metric == "SSIM":
            self.metric = ssim
        else:
            raise ValueError(f"got wrong metric name {metric}")

    def run_task(self):
        # prepare
        self.logger = get_logger("results", self.log_dir, level=20, file_mode="w")
        dump_cfg_yml(self.cfg, os.path.join(self.log_dir, "search_config.yml"))
        self.logger.log(20, "Searcher launched")
        self.img_dir = mkdirs(os.path.join(self.log_dir, "fp_imgs"))
        self.chr_dir = mkdirs(os.path.join(self.log_dir, "chromosomes"))
        self.manager = mp.Manager()
        self.eval_queue = self.manager.Queue()
        self.pred_queue = self.manager.Queue()  # microchannel haos coords and poses
        pred_pool = mp.Pool(self.cfg["num_pred_worker"])
        for _ in range(self.cfg["num_pred_worker"]):
            pred_pool.apply_async(
                func=self.tt_pred_worker,
                args=(self.pred_queue, self.eval_queue),
            )
        pred_pool.close()
        self.ranked_fitness = 1 / np.sqrt(np.arange(self.cfg["population"]) + 1)
        ### computation start
        self.logger.log(
            20,
            f"  {'Generation_id':10s},  {'Chr_id':>8s},  {'HAO_id':>8s},  {'Raw_fitness':>12s}",
        )
        # initialize a generation
        chromosomes = self.init_chr()  # saved current recipe to self. coords pos, pin.
        # run generations:
        top_raw_fitness_stats = {
            "mean": np.array([]),
            "std": np.array([]),
            "min": np.array([]),
            "max": np.array([]),
        }
        fig, ax = plt.subplots(figsize=(5, 3), dpi=600)
        for gen_id in range(self.cfg["max_generation"]):
            # compute ranked fitness
            self.gen_id = gen_id
            ranked_fitness, gen_top_rf_stats = self.run_fitness(chromosomes)
            chromosomes_new = []
            if self.num_elite > 0:
                chromosomes_new.append(self.run_elitism(chromosomes, ranked_fitness))
            offsprings = self.run_crossover(chromosomes, ranked_fitness)
            chromosomes_new.append(self.run_mutation(offsprings))
            chromosomes = np.concatenate(chromosomes_new, axis=0)
            # visualize progress
            for k, v in top_raw_fitness_stats.items():
                top_raw_fitness_stats[k] = np.append(v, gen_top_rf_stats[k])
            ax.cla()
            ax.fill_between(
                x=np.arange(gen_id + 1),
                y1=top_raw_fitness_stats["max"],
                y2=top_raw_fitness_stats["min"],
                lw=0,
                color="r",
                alpha=0.3,
            )
            ax.fill_between(
                x=np.arange(gen_id + 1),
                y1=top_raw_fitness_stats["mean"] - top_raw_fitness_stats["std"],
                y2=top_raw_fitness_stats["mean"] + top_raw_fitness_stats["std"],
                lw=1,
                color="b",
                alpha=0.3,
            )
            ax.plot(top_raw_fitness_stats["mean"], lw=2, color="black")
            ax.set_xlabel("Generations")
            ax.set_ylabel("Top raw fitnesses")
            # ax.set_title('search progress')
            fig.tight_layout()
            plt.savefig(os.path.join(self.log_dir, "progress.png"))
        ### the end ###
        [self.pred_queue.put("End") for _ in range(self.cfg["num_pred_worker"])]
        pred_pool.join()
        pass

    def init_chr(self):
        # sample chromosome: hao coord params, hao pose params, pin params
        n_hcp = (
            self.cfg["max_num_hao"] * (self.hcp_generator.coord_meta["num_seg"] + 1) * 3
        )  # num of HAO coordiantes param
        n_hpp = self.cfg["max_num_hao"]  # num of HAO position param, each hao has 1
        n_pinp = 2  # num of pin params, 1 for flow rate ratio and 1 for relative position in fp.
        chromosomes = self.rng.uniform(
            0, 1, (self.cfg["population"], n_pinp + n_hpp + n_hcp)
        )
        return chromosomes

    def gene_express(self, chromosomes: np.ndarray):
        """decode the chromosome  uniformly distributed rand vars to microfludic params"""
        pin_params = chromosomes[:, 0:2]
        hao_pos_params = chromosomes[:, 2 : self.cfg["max_num_hao"] + 2]
        hao_coord_params = chromosomes[:, self.cfg["max_num_hao"] + 2 :].reshape(
            self.cfg["population"], self.cfg["max_num_hao"], -1
        )
        pin_bounds = self.calc_pin_bound(pin_params)
        hao_poses = (hao_pos_params > 0.5).astype("bool")
        hao_coords = self.hcp_generator.param2coord(
            hao_coord_params, self.hcp_generator.coord_meta
        )
        msc = {
            "pin_bounds": pin_bounds,
            "hao_poses": hao_poses,
            "hao_coords": hao_coords,
        }
        return msc  # microfluidic system configuration

    def calc_pin_bound(self, pin_params: np.ndarray):
        # ZY currently only support single-color-stripe inlet profiles
        """
        pin params of shape (population, 2)
        """
        fr_ratio = (
            pin_params[..., 0] * (self.fr_range[1] - self.fr_range[0])
            + self.fr_range[0]
        )
        # fr_ratio = np.round(fr_ratio,6)
        _, _, w3 = self.pin_param_calc.calc_wr_ratio(
            fr_ratio, 1 - 2 * fr_ratio, fr_ratio
        )
        pin_l_bounds = pin_params[..., 1] * (1 - w3)  # shape (population,)
        pin_r_bounds = self.pin_param_calc.find_r_bound(pin_l_bounds, fr_ratio)
        pin_bounds = (
            (np.stack([pin_l_bounds, pin_r_bounds], axis=-1) * self.cfg["fp_res"][0])
            .round()
            .astype("int16")
        )[..., np.newaxis]
        return pin_bounds  # shape (population, 2, c=1)

    @staticmethod
    def tt_pred_worker(obs_param_queue, eval_queue):
        predictor = TTPredictor()
        while True:
            try:
                job_pacakge = obs_param_queue.get(timeout=1)
                if job_pacakge is None:
                    continue
                elif job_pacakge == "End":
                    break
                hao_coords = job_pacakge["msc"].pop("hao_coords")
                hao_poses = job_pacakge["msc"].pop("hao_poses")
                tts = predictor.predict_from_obs_param(hao_coords, hao_poses)
                job_pacakge["tts"] = tts
                eval_queue.put(job_pacakge)
            except Empty:
                continue

    @staticmethod
    def eval_worker(
        eval_queue,
    ):
        job_pkg = eval_queue.get()
        cumulative_tts = tt_synth(job_pkg["tts"], interm=True)
        pin = gen_pin_tensor(
            param=job_pkg["msc"]["pin_bounds"],
            fp_res=job_pkg["fp_res"],
            param_scale=False,
        ).squeeze(
            0
        )  # shape (fp_res, fp_res, num_stripes)
        pouts = p_transform(pin=pin, tts=cumulative_tts, full_p_records=True)[
            job_pkg["start_num_hao"] :
        ]  # shape (num_hao, fp_res, fp_res, num_stripes)
        # compute the metrics
        metric = job_pkg["metric"]
        raw_fitness = metric(
            [pouts, job_pkg["target_fp"]]
        )  # shape (max_num_hao-min_num_hao+1)
        return job_pkg["chr_id"], raw_fitness, pouts

    def build_jobs(self, chromosomes):
        mscs = self.gene_express(chromosomes)
        jobs = []
        for i in range(len(chromosomes)):
            msc = {k: v[i] for k, v in mscs.items()}
            jobs.append(
                {
                    "chr_id": i,
                    "fp_res": self.cfg["fp_res"],
                    "target_fp": self.tgt_fp,
                    "msc": msc,
                    "start_num_hao": self.cfg["min_num_hao"],
                    "metric": self.metric,
                }
            )
        return jobs

    def run_fitness(self, chromosomes):
        jobs = self.build_jobs(chromosomes)
        eval_pool = mp.Pool(self.cfg["num_eval_worker"])
        results = []
        with Timer():
            for job in jobs:
                self.pred_queue.put(job)
                results.append(
                    eval_pool.apply_async(
                        func=self.eval_worker,
                        args=(self.eval_queue,),
                    )
                )
            eval_pool.close()
            eval_pool.join()
        # collect results
        chr_ids, raw_fitnesses, pouts = zip(*(result.get() for result in results))
        chr_ids, raw_fitnesses, pouts = map(np.array, (chr_ids, raw_fitnesses, pouts))
        # raw fitness of shape (population, max num hao - min num hao)
        rf_best, ranked_pos = self.rank_raw_fitness(raw_fitnesses)
        best_rf_stats = {
            "max": rf_best.max(),
            "mean": rf_best.mean(),
            "std": rf_best.std(),
            "min": rf_best.min(),
        }
        _, rf_best_id = np.nonzero(raw_fitnesses == rf_best[..., np.newaxis])
        rf_best_hao = rf_best_id + self.cfg["min_num_hao"]
        # log gen best 3 fitness, save the chromosomes and the profiles.
        fig, ax = create_profile_figure(3, 3, 400)
        for pos in ranked_pos[:3]:
            chr_fname = f"chr-{self.gen_id}-{chr_ids[pos]}-{rf_best_hao[pos]}-{rf_best[pos]:.3f}"
            self.logger.log(
                20,
                f"  {self.gen_id:>10d},  {chr_ids[pos]:>8d},  {rf_best_hao[pos]:>8d},  {rf_best[pos]:>10.5f}",
            )

            np.savez(
                os.path.join(self.chr_dir, chr_fname + ".npz"), **(jobs[chr_ids[pos]])
            )
            plot_fp_tensor(pouts[pos, rf_best_id[pos]], ax)
            fig.savefig(os.path.join(self.img_dir, chr_fname + ".png"))
        plt.close(fig)
        # get the chromosome id of raw fitness from high to low, aligned with fitness score.
        sorted_chr_ids = chr_ids[ranked_pos]
        # get the order to sort ranked fitness in chromo id acending order
        chr_id_ascend_order_pos = np.argsort(sorted_chr_ids)
        ranked_fitness = self.ranked_fitness[chr_id_ascend_order_pos]
        return ranked_fitness, best_rf_stats

    def run_elitism(self, chromosomes: dict, fitness: np.ndarray):
        elite_ids = np.nonzero(fitness > self.ranked_fitness[self.num_elite])
        elites = chromosomes[elite_ids]
        return elites

    def run_crossover(self, chromosomes, fitness):
        # NOTE: num_offsprings = num parents.
        parent_ids = self.stochastic_tounament_select(
            self.num_offspring, fitness, self.cfg["prob_select"]
        )
        # NOTE: curent select implementation may reuslt in the same chr to be both mom
        # and dad, bu the chance is less than 0.0016, so not big problem.
        # We may issue a warning.
        check_id = parent_ids if parent_ids.shape[0] % 2 == 0 else parent_ids[:-1]
        check_id = check_id.reshape(2, -1)
        duplicates = np.argwhere(check_id[0] == check_id[1])
        if len(duplicates):
            warnings.warn(
                f"found at least {len(duplicates)} parent pairs containing same chromosomes!"
            )
        offsprings, of_min, of_max = self.laplace_crossover(chromosomes[parent_ids])
        offsprings = self.resample_chr(offsprings, of_min, of_max)
        return offsprings

    def run_mutation(
        self,
        chromosomes: dict,
    ):
        """implement gaussian mutation"""
        num_chr = chromosomes.shape[0]
        num_genes = chromosomes.shape[1]
        # select chromosomes to mutate
        mut_chr_id = self.repeat_choose(
            num_chr, self.num_mut_chr, 1, same_set=True
        ).squeeze()
        mut_chrs = chromosomes[mut_chr_id]  # shape (num_mut_chr, num_gene, *gene_shape)
        # select genes to mutate.
        num_mut_genes = round(self.cfg["gene_mutation_rate"] * num_genes)
        mut_gene_ids = self.repeat_choose(
            num_genes, num_mut_genes, self.num_mut_chr, same_set=False
        )  # (num_mut_chr, num_mut_genes)
        mutated, m_min, m_max = self.gaussian_mutator(mut_chrs[mut_gene_ids], self.rng)
        mutated = self.resample_chr(mutated, m_min, m_max)
        mut_chrs[mut_gene_ids] = mutated
        chromosomes[mut_chr_id] = mut_chrs
        return chromosomes

    def rank_raw_fitness(self, raw_fitness: np.ndarray):
        rf_best = raw_fitness.max(axis=-1)
        ranked_pos = np.argsort(
            rf_best, axis=0
        )  # the position of raw fitness score low to high
        ranked_pos = ranked_pos[::-1]  # the position of rawfitness score high to low
        return rf_best, ranked_pos

    def stochastic_tounament_select(
        self, num_winner, fitness_scores: np.ndarray, probablistic=True
    ):
        """return the chr_ids of the slected chromosomes"""
        # TODO select abosolute one.
        num_chr = len(fitness_scores)
        candidate_ids = self.repeat_choose(
            num_chr, self.cfg["tournament_size"], num_winner, same_set=True
        )
        cand_fits = fitness_scores[candidate_ids]
        if probablistic:
            cand_prob = cand_fits / cand_fits.sum(
                1, keepdims=True
            )  # probability of being select is proportional to its fit score
            selected = [
                self.rng.choice(self.cfg["tournament_size"], p=cand_prob[i])
                for i in range(num_winner)
            ]
        else:
            selected = np.argmax(cand_fits, axis=-1)
            pass
        return candidate_ids[np.arange(num_winner), selected]

    def laplace_crossover(self, parents_params: np.ndarray):
        """parents of shape (num_chr, num_gene, ...). parents order should already be random."""
        p_shape = parents_params.shape
        if p_shape[0] % 2 != 0:
            added_id = self.rng.integers(0, p_shape[0])
            parents_params = np.concatenate(
                [parents_params, parents_params[added_id][np.newaxis, ...]]
            )
        parents_params = parents_params.reshape(2, (p_shape[0] + 1) // 2, *p_shape[1:])
        distance = np.abs(parents_params[0] - parents_params[1])
        beta = laplace_sampler(
            distance.shape, miu=self.cfg["lap_miu"], b=self.cfg["lap_b"], rng=self.rng
        )  # laplace parameter copied from flow sculpt
        offsprings = parents_params + beta * distance  # shape(2, num_chr//2, ...)
        of_min = parents_params + self.lap_xo_beta_min * distance
        of_max = parents_params + self.lap_xo_beta_max * distance
        offsprings = offsprings.reshape(-1, *p_shape[1:])[
            : p_shape[0]
        ]  # shape (num_chr, ...)
        of_min = of_min.reshape(-1, *p_shape[1:])[: p_shape[0]]  # shape (num_chr, ...)
        of_max = of_max.reshape(-1, *p_shape[1:])[: p_shape[0]]  # shape (num_chr, ...)
        return offsprings, of_min, of_max

    def gaussian_mutator(self, x: np.ndarray, rng, x_interval_length=1):
        """mutation value range ~ [- half interval length, + half interval length]"""
        lower_bound = x - 0.5 * x_interval_length
        upper_bound = x + 0.5 * x_interval_length
        mag = rng.normal(0, self.cfg["gaussian_mut_scale"], x.shape) * x_interval_length
        x = x + mag
        return x, lower_bound, upper_bound

    def resample_chr(
        self,
        param: np.ndarray,
        param_min,
        param_max,
    ):
        """resample out of bound params into the intersaction of vliad range and modification range
        Args:
            param min, param max are the bound values of of the crossover or mutation.
        """
        # ensure the values in valid range.
        lower_bound = np.maximum(param_min, 0)
        upper_bound = np.minimum(param_max, 1)
        tbr_id = (param < lower_bound) + (param > upper_bound)
        resampled = self.rng.uniform(lower_bound, upper_bound)
        param[tbr_id] = resampled[tbr_id]
        return param

    def repeat_choose(self, num_choices, num_chosen, num_repeat, same_set: bool):
        """from a set of num_choices items, choose num_chosen items. the selection is
        repeated num_repeat times independently on the same set or on number of num_repeat
        sets.
        If choose from the same set, return a 2D array. If choose from different sets,
        return a tuple containing the set id and the corresponding chosen item id.
        used for selecting parents and mutated genes.
        """
        choice_ids = np.tile(np.arange(num_choices), reps=(num_repeat, 1))
        selected_ids = self.rng.permuted(choice_ids, axis=1)[:, :num_chosen]
        if not same_set:
            selected_ids = (
                np.arange(num_repeat)[..., np.newaxis],
                selected_ids,
            )
        return selected_ids
