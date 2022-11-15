import logging
import pickle
from typing import Callable, Dict, Iterable, List, OrderedDict

from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from analysis import MetricValueMeter, MultiClassAccuracyMeter
from dnf_layer import SemiSymbolicLayerType
from eval import asp_eval, dnf_eval
from rule_learner import DNFBasedClassifier, DNFClassifier, DNFClassifierEO
from train import (
    get_partial_cub_data_path_dict,
    loss_func_map,
)
from utils import get_dnf_classifier_x_and_y, load_partial_cub_data


log = logging.getLogger()


def prune_layer_weight(
    model: DNFBasedClassifier,
    layer_type: SemiSymbolicLayerType,
    epsilon: float,
    data_loader: DataLoader,
    use_cuda: bool,
    use_jaccard_meter: bool = False,
    show_tqdm: bool = False,
) -> int:
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = model.dnf.conjunctions.weights.data.clone()
    else:
        curr_weight = model.dnf.disjunctions.weights.data.clone()

    og_accuracy = dnf_eval(model, use_cuda, data_loader, use_jaccard_meter)

    prune_count = 0
    weight_device = curr_weight.device

    flatten_weight_len = len(torch.reshape(curr_weight, (-1,)))
    base_iterator = range(flatten_weight_len)
    iterator = tqdm(base_iterator) if show_tqdm else base_iterator

    for i in iterator:
        curr_weight_flatten = torch.reshape(curr_weight, (-1,))

        if curr_weight_flatten[i] == 0:
            continue

        mask = torch.ones(flatten_weight_len, device=weight_device)
        mask[i] = 0
        mask = mask.reshape(curr_weight.shape)

        masked_weight = curr_weight * mask

        if layer_type == SemiSymbolicLayerType.CONJUNCTION:
            model.dnf.conjunctions.weights.data = masked_weight
        else:
            model.dnf.disjunctions.weights.data = masked_weight

        new_perf = dnf_eval(model, use_cuda, data_loader, use_jaccard_meter)
        performance_drop = og_accuracy - new_perf
        if performance_drop < epsilon:
            prune_count += 1
            curr_weight *= mask

    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        model.dnf.conjunctions.weights.data = curr_weight
    else:
        model.dnf.disjunctions.weights.data = curr_weight
    return prune_count


def remove_unused_conjunctions(model: DNFBasedClassifier) -> int:
    disj_w = model.dnf.disjunctions.weights.data.clone()
    unused_count = 0

    for i, w in enumerate(disj_w.T):
        if torch.all(w == 0):
            # The conjunction is not used at all
            model.dnf.conjunctions.weights.data[i, :] = 0
            unused_count += 1

    return unused_count


def apply_threshold(
    model: DNFBasedClassifier,
    og_conj_weight: Tensor,
    og_disj_weight: Tensor,
    t_val: Tensor,
    const: float = 6.0,
) -> None:
    new_conj_weight = (
        (torch.abs(og_conj_weight) > t_val) * torch.sign(og_conj_weight) * const
    )
    model.dnf.conjunctions.weights.data = new_conj_weight

    new_disj_weight = (
        (torch.abs(og_disj_weight) > t_val) * torch.sign(og_disj_weight) * const
    )
    model.dnf.disjunctions.weights.data = new_disj_weight


def extract_asp_rules(sd: OrderedDict, flatten: bool = False) -> List[str]:
    output_rules = []

    # Get all conjunctions
    conj_w = sd["dnf.conjunctions.weights"]
    conjunction_map = dict()
    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # No conjunction is applied here
            continue

        conjuncts = []
        for j, v in enumerate(w):
            if v < 0:
                # Negative weight, negate the atom
                conjuncts.append(f"not has_attr_{j}")
            elif v > 0:
                # Positive weight, normal atom
                conjuncts.append(f"has_attr_{j}")

        conjunction_map[i] = conjuncts

    if not flatten:
        # Add conjunctions as auxilary predicates into final rules list
        # if not flatten
        for k, v in conjunction_map.items():
            output_rules.append(f"conj_{k} :- {', '.join(v)}.")

    # Get DNF
    disj_w = sd["dnf.disjunctions.weights"]
    not_covered_classes = []
    for i, w in enumerate(disj_w):
        if torch.all(w == 0):
            # No DNF for class i
            not_covered_classes.append(i)
            continue

        disjuncts = []
        for j, v in enumerate(w):
            if v < 0 and j in conjunction_map:
                # Negative weight, negate the existing conjunction
                if flatten:
                    # Need to add auxilary predicate (conj_X) which is not yet
                    # in the final rules list
                    output_rules.append(
                        f"conj_{j} :- {', '.join(conjunction_map[j])}."
                    )
                    output_rules.append(f"class({i}) :- not conj_{j}.")
                else:
                    disjuncts.append(f"not conj_{j}")
            elif v > 0 and j in conjunction_map:
                # Postivie weight, add normal conjunction
                if flatten:
                    body = ", ".join(conjunction_map[j])
                    output_rules.append(f"class({i}) :- {body}.")
                else:
                    disjuncts.append(f"conj_{j}")

        if not flatten:
            for disjunct in disjuncts:
                output_rules.append(f"class({i}) :- {disjunct}.")

    return output_rules


class DNFBasedPostTrainingProcessor:
    # Data loaders
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    test_pkl_path: str

    # Post-training process parameters
    use_cuda: bool
    experiment_name: str
    optimiser_key: str
    optimiser_fn: Callable[[Iterable], Optimizer]
    loss_func_key: str
    criterion: Callable[[Tensor, Tensor], Tensor]
    reg_fn: str
    reg_lambda: float
    is_eo_based: bool  # init by child classes
    pth_file_base_name: str

    # Configs
    cfg: DictConfig
    model_train_cfg: DictConfig

    # Post-training process parameters
    prune_epsilon: float = 0.005
    tune_epochs: int = 100
    tune_weight_constraint_lambda: float = 0.1

    # Result info dictionary
    result_dict: Dict[str, float] = dict()

    def __init__(self, model_name: str, cfg: DictConfig) -> None:
        # Configs
        self.cfg = cfg
        self.model_train_cfg = cfg["training"][model_name]

        # Parameters
        self.use_cuda = (
            cfg["training"]["use_cuda"] and torch.cuda.is_available()
        )
        self.experiment_name = cfg["training"]["experiment_name"]

        random_seed = cfg["training"]["random_seed"]
        self.pth_file_base_name = (
            f"{self.experiment_name}_{random_seed}" + "_pd"
            if self.is_eo_based
            else ""
        )

        # Data loaders
        env_cfg = cfg["environment"]
        partial_cub_cfg = cfg["training"]["partial_cub"]
        batch_size = self.model_train_cfg["batch_size"]

        # Use existing partial pkl files
        self.train_loader, self.val_loader = load_partial_cub_data(
            is_training=True,
            batch_size=batch_size,
            data_path_dict=get_partial_cub_data_path_dict(
                env_cfg, partial_cub_cfg
            ),
            use_img_tensor=False,
        )
        self.test_loader = load_partial_cub_data(
            is_training=False,
            batch_size=batch_size,
            data_path_dict=get_partial_cub_data_path_dict(
                env_cfg, partial_cub_cfg
            ),
            use_img_tensor=False,
        )

        self.test_pkl_path = partial_cub_cfg["partial_test_pkl"]

        # Tuning optimiser
        lr = self.model_train_cfg["optimiser_lr"]
        weight_decay = self.model_train_cfg["optimiser_weight_decay"]
        self.optimiser_key = self.model_train_cfg["optimiser"]
        if self.optimiser_key == "sgd":
            self.optimiser_fn = lambda params: torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            self.optimiser_fn = lambda params: torch.optim.Adam(
                params, lr=lr, weight_decay=weight_decay
            )

        # Tuning loss function
        self.loss_func_key = self.model_train_cfg["loss_func"]
        self.criterion = loss_func_map[self.loss_func_key]

        # Other parameters
        self.reg_fn = self.model_train_cfg["reg_fn"]
        self.reg_lambda = self.model_train_cfg["reg_lambda"]

    def _after_train_eval(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def _pruning(self, model: DNFBasedClassifier) -> None:
        if self.is_eo_based:
            log.info("Pruning on plain DNF of DNF-EO starts")
        else:
            log.info("Pruning on DNF starts")

        log.info("Prune disj layer")
        prune_count = prune_layer_weight(
            model,
            SemiSymbolicLayerType.DISJUNCTION,
            self.prune_epsilon,
            self.val_loader,
            self.use_cuda,
            use_jaccard_meter=True,
        )
        new_perf = dnf_eval(model, self.use_cuda, self.val_loader, True)
        log.info(f"Pruned disj count:   {prune_count}")
        log.info(f"New perf after disj: {new_perf:.3f}")

        unused_conj = remove_unused_conjunctions(model)
        log.info(f"Remove unused conjunctios: {unused_conj}")

        log.info("Prune conj layer")
        prune_count = prune_layer_weight(
            model,
            SemiSymbolicLayerType.CONJUNCTION,
            self.prune_epsilon,
            self.val_loader,
            self.use_cuda,
            use_jaccard_meter=True,
        )
        new_perf = dnf_eval(model, self.use_cuda, self.test_loader, True)
        log.info(f"Pruned conj count:   {prune_count}")
        log.info(f"New perf after conj: {new_perf:.3f}\n")

        torch.save(model.state_dict(), self.pth_file_base_name + "_pruned.pth")
        self.result_dict["after_prune"] = round(new_perf, 3)

    def _tuning(self, model: DNFBasedClassifier) -> None:
        log.info(f"Tuning of {'DNF-EO' if self.is_eo_based else 'DNF'} start")

        initial_cjw = model.dnf.conjunctions.weights.data.clone()
        initial_djw = model.dnf.disjunctions.weights.data.clone()

        cjw_mask = torch.where(initial_cjw != 0, 1, 0)
        djw_mask = torch.where(initial_djw != 0, 1, 0)

        cjw_inverse_mask = torch.where(initial_cjw != 0, 0, 1)
        djw_inverse_mask = torch.where(initial_djw != 0, 0, 1)

        weight_device = initial_cjw.device

        model.conj_weight_mask = cjw_mask.to(weight_device)
        model.disj_weight_mask = djw_mask.to(weight_device)

        # Weight pushing loss
        def dnf_weight_pushing_constraint():
            # The loss should be only applied to not pruned weights
            conj_non_zero_w = torch.masked_select(
                model.dnf.conjunctions.weights.data,
                model.conj_weight_mask.bool(),
            )
            disj_non_zero_w = torch.masked_select(
                model.dnf.disjunctions.weights.data,
                model.disj_weight_mask.bool(),
            )

            def _constraint(w):
                # Pushing the weight to 6/-6/0
                # w * |6 - |w||
                return torch.abs(w * (6 - torch.abs(w))).sum()

            return _constraint(conj_non_zero_w) + _constraint(disj_non_zero_w)

        # Other setup
        optimizer = self.optimiser_fn(model.parameters())

        for epoch in range(self.tune_epochs):
            loss_meter = MetricValueMeter("loss")
            perf_meter = MultiClassAccuracyMeter()

            for data in self.train_loader:
                assert torch.all(
                    torch.masked_select(
                        model.dnf.conjunctions.weights.data,
                        cjw_inverse_mask.bool().to(weight_device),
                    )
                    == 0
                )
                assert torch.all(
                    torch.masked_select(
                        model.dnf.disjunctions.weights.data,
                        djw_inverse_mask.bool().to(weight_device),
                    )
                    == 0
                )

                optimizer.zero_grad()
                x, y = get_dnf_classifier_x_and_y(data, self.use_cuda)
                y_hat = model(x)

                wc = dnf_weight_pushing_constraint()
                loss = (
                    1 - self.tune_weight_constraint_lambda
                ) * self.criterion(
                    y_hat, y
                ) + self.tune_weight_constraint_lambda * wc

                loss.backward()
                optimizer.step()

                # Maintain the pruned weights stay as 0
                model.update_weight_wrt_mask()

                loss_meter.update(loss.item())
                perf_meter.update(y_hat, y)

            log.info(
                "[%3d] Finetune  avg loss: %.3f  avg perf: %.3f"
                % (
                    epoch + 1,
                    loss_meter.get_average(),
                    perf_meter.get_average(),
                )
            )

        acc = dnf_eval(model, self.use_cuda, self.test_loader)
        jacc = dnf_eval(
            model, self.use_cuda, self.test_loader, use_jaccard_meter=True
        )
        log.info(f"Accuracy after tune: {acc:.3f}")
        log.info(f"Jaccard after tune:  {jacc:.3f}\n")

        torch.save(model.state_dict(), self.pth_file_base_name + "_tuned.pth")

        self.result_dict["after_tune_acc"] = round(acc, 3)
        self.result_dict["after_tune_jacc"] = round(jacc, 3)

    def _thresholding(self, model: DNFBasedClassifier) -> None:
        if self.is_eo_based:
            log.info("Thresholding on plain DNF of DNF-EO starts")
        else:
            log.info("Thresholding on DNF starts")

        conj_min = torch.min(model.dnf.conjunctions.weights.data)
        conj_max = torch.max(model.dnf.conjunctions.weights.data)
        disj_min = torch.min(model.dnf.disjunctions.weights.data)
        disj_max = torch.max(model.dnf.disjunctions.weights.data)

        threshold_upper_bound = round(
            (
                torch.Tensor([conj_min, conj_max, disj_min, disj_max])
                .abs()
                .max()
                + 0.01
            ).item(),
            2,
        )

        og_conj_weight = model.dnf.conjunctions.weights.data.clone()
        og_disj_weight = model.dnf.disjunctions.weights.data.clone()

        jacc_scores = []
        t_vals = torch.arange(0, threshold_upper_bound, 0.01)

        for v in t_vals:
            apply_threshold(model, og_conj_weight, og_disj_weight, v, 6.0)
            jacc = dnf_eval(model, self.use_cuda, self.val_loader, True)
            jacc_scores.append(jacc)

        best_jacc_score = max(jacc_scores)
        best_t = t_vals[torch.argmax(torch.Tensor(jacc_scores))]
        log.info(
            f"Best t: {best_t.item():.3f}    Jaccard: {best_jacc_score:.3f}"
        )

        apply_threshold(model, og_conj_weight, og_disj_weight, best_t)

        val_jaccard = dnf_eval(model, self.use_cuda, self.val_loader, True)
        test_jaccard = dnf_eval(model, self.use_cuda, self.test_loader, True)
        log.info(f"Val jacc after threshold:  {val_jaccard:.3f}")
        log.info(f"Test jacc after threshold: {test_jaccard:.3f}\n")

        torch.save(
            model.state_dict(), self.pth_file_base_name + "_thresholded.pth"
        )

        self.result_dict["after_threshold_val_jacc"] = round(val_jaccard, 3)
        self.result_dict["after_threshold_test_jacc"] = round(test_jaccard, 3)

    def _extract_rules(self, model: DNFBasedClassifier) -> None:
        log.info("Rule extraction starts")
        log.info("Rules:")
        rules = extract_asp_rules(model.state_dict(), flatten=True)
        for r in rules:
            log.info(r)

        with open(self.test_pkl_path, "rb") as f:
            test_data = pickle.load(f)

        acc, avg_jacc_score = asp_eval(test_data, rules, debug=False)

        log.info("Extracted rules result:")
        log.info(f"Accuracy:   {acc:.3f}")
        log.info(f"Jacc score: {avg_jacc_score:.3f}")

        with open(self.pth_file_base_name + "_rules.txt", "w") as f:
            f.write("\n".join(rules))

        self.result_dict["rule_acc"] = round(acc, 3)
        self.result_dict["rule_jacc"] = round(avg_jacc_score, 3)

    def post_processing(self, model: DNFBasedClassifier) -> Dict[str, float]:
        raise NotImplementedError


class VanillaDNFPostTrainingProcessor(DNFBasedPostTrainingProcessor):
    is_eo_based: bool = False

    def __init__(self, model_name: str, cfg: DictConfig) -> None:
        super().__init__(model_name, cfg)

    def _after_train_eval(self, model: DNFClassifier) -> None:
        log.info("DNF performance after train")

        acc = dnf_eval(model, self.use_cuda, self.test_loader)
        jacc = dnf_eval(model, self.use_cuda, self.test_loader, True)

        log.info(f"DNF  accuracy: {acc:.3f}  jaccard: {jacc:.3f}\n")

        self.result_dict["after_train_acc"] = round(acc, 3)
        self.result_dict["after_train_jacc"] = round(jacc, 3)

    def post_processing(self, model: DNFClassifier) -> Dict[str, float]:
        log.info("\n------- Post Processing -------")

        self._after_train_eval(model)
        self._pruning(model)
        self._tuning(model)
        self._thresholding(model)
        self._extract_rules(model)

        return self.result_dict


class DNFEOPostTrainingProcessor(DNFBasedPostTrainingProcessor):
    is_eo_based: bool = True

    def __init__(self, model_name: str, cfg: DictConfig) -> None:
        super().__init__(model_name, cfg)

    def _after_train_eval(
        self, model: DNFClassifierEO, model2: DNFClassifier
    ) -> None:
        log.info("DNF-EO and its plain DNF performance after train")

        acc = dnf_eval(model, self.use_cuda, self.test_loader)
        jacc = dnf_eval(model, self.use_cuda, self.test_loader, True)

        acc_2 = dnf_eval(model2, self.use_cuda, self.test_loader)
        jacc_2 = dnf_eval(model2, self.use_cuda, self.test_loader, True)

        log.info(f"DNF-EO accuracy: {acc:.3f}  jaccard: {jacc:.3f}")
        log.info(f"DNF    accuracy: {acc_2:.3f}  jaccard: {jacc_2:.3f}\n")

        self.result_dict["after_train_acc"] = round(acc, 3)
        self.result_dict["after_train_jacc"] = round(jacc, 3)
        self.result_dict["pd_after_train_acc"] = round(acc_2, 3)
        self.result_dict["pd_after_train_jacc"] = round(jacc_2, 3)

    def post_processing(self, model: DNFClassifierEO) -> Dict[str, float]:
        log.info("\n ------- Post Processing -------")

        # Prepare plain DNF of DNF-EO
        num_preds = model.dnf.conjunctions.weights.shape[1]
        num_conjuncts = model.dnf.conjunctions.weights.shape[0]
        num_classes = model.dnf.disjunctions.weights.shape[0]
        model2 = DNFClassifier(num_preds, num_conjuncts, num_classes)

        sd = model.state_dict()
        sd.pop("eo_layer.weights")
        model2.load_state_dict(sd)

        #    After train eval
        self._after_train_eval(model, model2)

        # 1. Prune only plain DNF
        self._pruning(model2)

        # 2. Tune on DNF-EO
        model.load_dnf_state_dict(model2.state_dict())
        self._tuning(model)

        #    Extra eval of plain DNF of DNF-EO after tuning
        tune_sd = model.state_dict()
        tune_sd.pop("eo_layer.weights")
        model2.load_state_dict(tune_sd)

        log.info("Plain DNF of DNF-EO performance after tune")
        acc = dnf_eval(model2, self.use_cuda, self.test_loader)
        jacc = dnf_eval(model2, self.use_cuda, self.test_loader, True)
        log.info(f"Plain DNF   accuracy: {acc:.3f}  jaccard: {jacc:.3f}\n")
        self.result_dict["pd_after_tune_acc"] = round(acc, 3)
        self.result_dict["pd_after_tune_jacc"] = round(jacc, 3)

        #    Overwrite the pth of DNF-EO with plain DNF
        torch.save(model2.state_dict(), self.pth_file_base_name + "_tuned.pth")

        # 3. Threshold on plain DNF
        self._thresholding(model2)

        # 4. Extract rules from plain DNF
        self._extract_rules(model2)

        return self.result_dict
