from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ecp.utils import (  # noqa: E402
    QUANTIFIER_OPTION_DEFAULT,
    QUANTIFIER_OPTION_KEY,
    SET_OPTION_HEADER,
    canonical_structural_options_suffix,
    ecp_preamble,
    run_lean_code,
)


DATASETS = {
    "matharena": PROJECT_ROOT / "data/dataset/matharena.json",
    "putnam": PROJECT_ROOT / "data/dataset/putnam.json",
}


@dataclass(frozen=True)
class TestCase:
    dataset: str
    problem: str
    label: str
    candidate_answer: str
    expected_admissible: bool


TEST_CASES = (
    # Patrick, Tanya, and Jose travel from school to park at related constant speeds and arrive together; the distance is m/n miles, and the answer is m + n.
    TestCase("matharena", "P2026AIMEI_1", "literal_answer", "277", True),
    TestCase("matharena", "P2026AIMEI_1", "disguised_addition", "(276 : Nat) + 1", False),
    TestCase("matharena", "P2026AIMEI_1", "successor_form", "Nat.succ 276", False),

    # In an isosceles triangle ABC with incenter I, the perimeters of ABC and AIC have ratio 125:6; find the minimum possible integer side AB.
    TestCase("matharena", "P2026AIMEII_8", "literal_answer", "245", True),
    TestCase("matharena", "P2026AIMEII_8", "power_expression", "5 ^ 3 + 120", False),
    TestCase("matharena", "P2026AIMEII_8", "set_infimum_expression", "sInf ({245} : Set Nat)", False),

    # A line meets y = x^2 + 2/x in three points; two x-coordinates are 6 and 7, and the answer is the third x-coordinate.
    TestCase("matharena", "P2026HMMT_1", "signed_rational", "(-1 : ℝ) / 21", True),
    TestCase("matharena", "P2026HMMT_1", "sqrt_arithmetic", "Real.sqrt 2 / 7", True),
    TestCase("matharena", "P2026HMMT_1", "forbidden_sine", "Real.sin 1", False),
    TestCase("matharena", "P2026HMMT_1", "forbidden_factorial_cast", "(Nat.factorial 5 : ℝ)", False),

    # Pairwise distinct complex numbers a, b, c satisfy three quadratic symmetric equations; compute a.
    TestCase("matharena", "P2026HMMT_4", "complex_rational", "(7 : ℂ) / 2", True),
    TestCase("matharena", "P2026HMMT_4", "explicit_complex_point", "Complex.mk 1 2", True),
    TestCase("matharena", "P2026HMMT_4", "imaginary_unit_expression", "Complex.I + 1", True),
    TestCase("matharena", "P2026HMMT_4", "complex_exponential", "Complex.exp Complex.I", False),

    # Complex numbers alpha and beta satisfy alpha * beta + alpha + beta + 100 = 0 and have common modulus M; determine all possible M.
    TestCase("matharena", "P2026HMMT_32", "singleton_answer", "{10}", True),
    TestCase("matharena", "P2026HMMT_32", "closed_interval", "Set.Icc (0 : ℝ) 1", True),
    TestCase("matharena", "P2026HMMT_32", "whole_real_line", "Set.univ", True),
    TestCase("matharena", "P2026HMMT_32", "set_builder_singleton", "{x : ℝ | x = 10}", False),

    # In a triangle with ordered angles and external angle bisectors meeting opposite lines, AP = BQ = AB; find angle CAB.
    TestCase("putnam", "putnam_1965_a1", "pi_fraction", "Real.pi / 15", True),
    TestCase("putnam", "putnam_1965_a1", "trig_combination", "Real.sin 1 + Real.cos 1", True),
    TestCase("putnam", "putnam_1965_a1", "finite_sum_scalar", "(Finset.range 3).sum (fun n => (n : ℝ))", False),

    # Evaluate in closed form the binomial sum sum_{k=1}^n binomial(n,k) * k^2.
    TestCase(
        "putnam",
        "putnam_1962_a5",
        "closed_form",
        "fun n : ℕ => n * (n + 1) * 2^(n - 2)",
        True,
    ),
    TestCase("putnam", "putnam_1962_a5", "factorial_function", "fun n : ℕ => Nat.factorial n", True),
    TestCase(
        "putnam",
        "putnam_1962_a5",
        "finset_sum_function",
        "fun n : ℕ => (Finset.range n).sum (fun k => k)",
        False,
    ),

    # Find the longest run of equal nonzero decimal digits with which a perfect square can end, and the smallest square attaining it.
    TestCase("putnam", "putnam_1970_a3", "given_pair", "(3, 1444)", True),
    TestCase(
        "putnam",
        "putnam_1970_a3",
        "computed_pair",
        "(Nat.gcd 12 18, Nat.factorial 4)",
        True,
    ),
    TestCase(
        "putnam",
        "putnam_1970_a3",
        "summed_first_coordinate",
        "((Finset.range 4).sum (fun k => k), 1)",
        False,
    ),

    # Determine all supercontinuous functions f : [0,1] -> R, where Cesaro convergence of x_n implies Cesaro convergence of f(x_n).
    TestCase(
        "putnam",
        "putnam_1972_a3",
        "affine_family",
        "{f | ∃ A B : ℝ, ∀ x ∈ Set.Icc 0 1, f x = A * x + B}",
        True,
    ),
    TestCase("putnam", "putnam_1972_a3", "universal_set", "Set.univ", True),
    TestCase(
        "putnam",
        "putnam_1972_a3",
        "quantify_over_function",
        "{f | ∃ g : ℝ → ℝ, ∀ x : ℝ, f x = g x}",
        False,
    ),

    # Let S = {2^m 3^n : m,n integers} and P be the positive reals; decide whether S is dense in P.
    TestCase("putnam", "putnam_1963_b2", "true_literal", "True", True),
    TestCase("putnam", "putnam_1963_b2", "false_literal", "False", True),
    TestCase("putnam", "putnam_1963_b2", "conjunction_shape", "True ∧ True", False),
    TestCase("putnam", "putnam_1963_b2", "numeric_equality", "(0 : ℕ) = 0", False),

    # For n = 2m with odd m > 1 and theta = exp(2*pi*i/n), express (1 - theta)^-1 as an integer polynomial in theta.
    TestCase(
        "putnam",
        "putnam_1975_a4",
        "odd_monomial_sum",
        "fun m => ∑ j ∈ Finset.range ((m - 1) / 2), Polynomial.X ^ (2 * j + 1)",
        True,
    ),
    TestCase(
        "putnam",
        "putnam_1975_a4",
        "constant_polynomial",
        "fun n : ℕ => Polynomial.C (n : ℤ)",
        True,
    ),
    TestCase(
        "putnam",
        "putnam_1975_a4",
        "derivative_formula",
        "fun n : ℕ => Polynomial.derivative (Polynomial.X ^ n : Polynomial ℤ)",
        False,
    ),

    # Find all +/-1-coefficient polynomials whose roots are all real.
    TestCase(
        "putnam",
        "putnam_1968_a6",
        "full_solution_set",
        "{X - 1, -(X - 1), X + 1, -(X + 1), X^2 + X - 1, -(X^2 + X - 1), "
        "X^2 - X - 1, -(X^2 - X - 1), X^3 + X^2 - X - 1, "
        "-(X^3 + X^2 - X - 1), X^3 - X^2 - X + 1, -(X^3 - X^2 - X + 1)}",
        True,
    ),
    TestCase("putnam", "putnam_1968_a6", "singleton_polynomial", "({Polynomial.X} : Set ℂ[X])", True),
    TestCase(
        "putnam",
        "putnam_1968_a6",
        "two_signed_polynomials",
        "({Polynomial.X, -Polynomial.X} : Set ℂ[X])",
        True,
    ),
    TestCase("putnam", "putnam_1968_a6", "polynomial_universe", "(Set.univ : Set ℂ[X])", True),
    TestCase(
        "putnam",
        "putnam_1968_a6",
        "polynomial_set_builder",
        "{p : ℂ[X] | p = Polynomial.X}",
        False,
    ),
)


def load_rows_by_name() -> Dict[str, Dict[str, Dict[str, Any]]]:
    rows_by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset, path in DATASETS.items():
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows_by_dataset[dataset] = {
            row["name"]: row
            for row in rows
            if isinstance(row, dict) and row.get("name")
        }
    return rows_by_dataset


def bool_from_metadata(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return default


def admissibility_metadata(row: Dict[str, Any]) -> Tuple[str, bool]:
    info = row.get("formal_answer_info")
    if not isinstance(info, dict):
        info = {}
    admissible_vocabulary = str(info.get("admissible_vocabulary") or "[]").strip()
    allow_quantifier = bool_from_metadata(
        info.get(QUANTIFIER_OPTION_KEY),
        QUANTIFIER_OPTION_DEFAULT,
    )
    return admissible_vocabulary, allow_quantifier


def checker_info(row: Dict[str, Any]) -> Dict[str, Any]:
    admissible_vocabulary, allow_quantifier = admissibility_metadata(row)
    return {
        "header": ecp_preamble(row),
        "answer_type": row["answer_type"],
        "admissible_vocabulary": admissible_vocabulary,
        QUANTIFIER_OPTION_KEY: allow_quantifier,
    }


def grouped_test_cases() -> Dict[Tuple[str, str], List[TestCase]]:
    grouped: Dict[Tuple[str, str], List[TestCase]] = {}
    for case in TEST_CASES:
        grouped.setdefault((case.dataset, case.problem), []).append(case)
    return grouped


def validate_test_suite() -> None:
    grouped = grouped_test_cases()
    if len(grouped) < 10:
        raise ValueError(f"Expected at least 10 problems, found {len(grouped)}.")
    for (dataset, problem), cases in grouped.items():
        if len(cases) < 3:
            raise ValueError(f"{dataset}:{problem} has fewer than 3 candidate answers.")
        if not any(case.expected_admissible for case in cases):
            raise ValueError(f"{dataset}:{problem} has no expected-admissible candidate.")
        if not any(not case.expected_admissible for case in cases):
            raise ValueError(f"{dataset}:{problem} has no expected-inadmissible candidate.")


def vocabulary_size(vocabulary: str) -> int:
    return len(re.findall(r"``[^,\]\s]+", vocabulary))


def one_line(text: Any, limit: int = 100) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def extract_canonical_statuses(result: Any) -> List[bool]:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    blocks = [
        block.strip().rstrip().replace("\n", "")
        for block in re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    ]
    return [
        block == "canonical"
        for block in blocks
        if block in {"canonical", "not canonical"}
    ]


def check_cases_for_problem(cases: List[TestCase], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    info = checker_info(row)
    structural = canonical_structural_options_suffix(info)
    commands = "\n".join(
        f"#isCanonical ({case.candidate_answer} : {info['answer_type']}) "
        f"with admissible_vocabulary := {info['admissible_vocabulary']} {structural}"
        for case in cases
    )
    lean_code = (
        "import utils.canonical_all_in_one\n"
        f"{info['header']}\n"
        f"{SET_OPTION_HEADER}\n"
        f"{commands}\n"
    )
    statuses = extract_canonical_statuses(run_lean_code(lean_code))
    status_count_matches = len(statuses) == len(cases)
    results = []
    for index, case in enumerate(cases):
        actual_admissible = statuses[index] if index < len(statuses) else False
        results.append(
            {
                "dataset": case.dataset,
                "problem": case.problem,
                "label": case.label,
                "candidate_answer": case.candidate_answer,
                "actual_admissible": bool(actual_admissible),
                "expected_admissible": case.expected_admissible,
                "passed": status_count_matches
                and bool(actual_admissible) == case.expected_admissible,
                "status_count_matches": status_count_matches,
            }
        )
    return results


def print_problem_header(row: Dict[str, Any], dataset: str, problem: str) -> None:
    admissible_vocabulary, allow_quantifier = admissibility_metadata(row)
    print(f"{dataset}:{problem}", flush=True)
    print(f"  answer_type: {row['answer_type']}", flush=True)
    print(f"  {QUANTIFIER_OPTION_KEY}: {str(allow_quantifier).lower()}", flush=True)
    print(
        f"  admissible_vocabulary_size: {vocabulary_size(admissible_vocabulary)}",
        flush=True,
    )


def print_case_result(result: Dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"  {status:<4} {result['label']:<24} "
        f"expected={str(result['expected_admissible']):<5} "
        f"actual={str(result['actual_admissible']):<5} "
        f"answer={one_line(result['candidate_answer'])}",
        flush=True,
    )


def main() -> int:
    validate_test_suite()
    rows_by_name = load_rows_by_name()
    grouped = grouped_test_cases()
    results = []

    print("Admissibility checker dataset smoke tests", flush=True)
    print("Metadata used: formal_answer_info.admissible_vocabulary", flush=True)
    print(f"Structural option used: formal_answer_info.{QUANTIFIER_OPTION_KEY}", flush=True)
    print("Note: this checks Lean syntax plus canonical admissibility, not answer correctness.")
    print(f"Suite shape: {len(grouped)} problems, {len(TEST_CASES)} candidate answers")
    print(flush=True)

    for (dataset, problem), cases in grouped.items():
        row = rows_by_name[dataset][problem]
        print_problem_header(row, dataset, problem)
        problem_results = check_cases_for_problem(cases, row)
        results.extend(problem_results)
        for result in problem_results:
            print_case_result(result)

    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    print(flush=True)
    print(f"Summary: {passed}/{total} passed", flush=True)
    if passed != total:
        print("Failed cases:", flush=True)
        for result in results:
            if not result["passed"]:
                status_note = ""
                if not result.get("status_count_matches", True):
                    status_note = " status_count_mismatch=True"
                print(
                    f"  {result['dataset']}:{result['problem']} {result['label']} "
                    f"expected={result['expected_admissible']} "
                    f"actual={result['actual_admissible']}{status_note}",
                    flush=True,
                )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
