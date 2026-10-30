#!/usr/bin/env python3
"""Test script for MultiPL-E evaluation with hardcoded outputs."""

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from datasets import load_dataset
from lm_eval.metrics.multiple_metrics.containerized_eval import eval_string_script
from lm_eval.tasks.multiple.multiple_utils import build_program, process_resp


TESTS = {
    "js": {
        "correct": """
    for (let i = 0; i < numbers.length; i++) {
        for (let j = i + 1; j < numbers.length; j++) {
            if (Math.abs(numbers[i] - numbers[j]) < threshold) {
                return true;
            }
        }
    }
    return false;
}
""",
        "wrong": """
    return false;
}
"""
    },
    "rs": {
        "correct": """
    for i in 0..numbers.len() {
        for j in (i+1)..numbers.len() {
            if (numbers[i] - numbers[j]).abs() < threshold {
                return true;
            }
        }
    }
    false
""",
        "wrong": """
    false
"""
    },
    "go": {
        "correct": """
	for i := 0; i < len(numbers); i++ {
		for j := i + 1; j < len(numbers); j++ {
			diff := numbers[i] - numbers[j]
			if diff < 0 {
				diff = -diff
			}
			if diff < threshold {
				return true
			}
		}
	}
	return false
}
""",
        "wrong": """
	return false
}
"""
    },
    "lua": {
        "correct": """
    for i = 1, #numbers do
        for j = i + 1, #numbers do
            if math.abs(numbers[i] - numbers[j]) < threshold then
                return true
            end
        end
    end
    return false
end
""",
        "wrong": """
    return false
end
"""
    },
    "rb": {
        "correct": """
  numbers.each_with_index do |n1, i|
    numbers.drop(i + 1).each do |n2|
      return true if (n1 - n2).abs < threshold
    end
  end
  false
end
""",
        "wrong": """
  false
end
"""
    },
    "cpp": {
        "correct": """
    for (int i = 0; i < numbers.size(); i++) {
        for (int j = i + 1; j < numbers.size(); j++) {
            if (std::abs(numbers[i] - numbers[j]) < threshold) {
                return true;
            }
        }
    }
    return false;
""",
        "wrong": """
    return false;
"""
    },
    "java": {
        "correct": """
        for (int i = 0; i < numbers.size(); i++) {
            for (int j = i + 1; j < numbers.size(); j++) {
                if (Math.abs(numbers.get(i) - numbers.get(j)) < threshold) {
                    return true;
                }
            }
        }
        return false;
""",
        "wrong": """
        return false;
"""
    },
    "php": {
        "correct": """
    $n = count($numbers);
    for ($i = 0; $i < $n; $i++) {
        for ($j = $i + 1; $j < $n; $j++) {
            if (abs($numbers[$i] - $numbers[$j]) < $threshold) {
                return true;
            }
        }
    }
    return false;
}
""",
        "wrong": """
    return false;
}
"""
    },
    "pl": {
        "correct": """
    my @nums = @{$numbers};
    my $n = scalar @nums;
    for my $i (0..$n-1) {
        for my $j ($i+1..$n-1) {
            if (abs($nums[$i] - $nums[$j]) < $threshold) {
                return 1;
            }
        }
    }
    return 0;
}
""",
        "wrong": """
    return 0;
}
"""
    },
    "sh": {
        "correct": """
    local numbers=($1)
    local threshold=$2
    local n=${#numbers[@]}
    for ((i=0; i<n; i++)); do
        for ((j=i+1; j<n; j++)); do
            local diff=$(echo "${numbers[i]} - ${numbers[j]}" | bc -l)
            if (( $(echo "$diff < 0" | bc -l) )); then
                diff=$(echo "$diff * -1" | bc -l)
            fi
            if (( $(echo "$diff < $threshold" | bc -l) )); then
                echo "true"
                return
            fi
        done
    done
    echo "false"
""",
        "wrong": """
    echo "false"
"""
    },
    "ts": {
        "correct": """
    for (let i = 0; i < numbers.length; i++) {
        for (let j = i + 1; j < numbers.length; j++) {
            if (Math.abs(numbers[i] - numbers[j]) < threshold) {
                return true;
            }
        }
    }
    return false;
}
""",
        "wrong": """
    return false;
}
"""
    },
    "go": {
        "correct": """
	for i := 0; i < len(numbers); i++ {
		for j := i + 1; j < len(numbers); j++ {
			diff := numbers[i] - numbers[j]
			if diff < 0 {
				diff = -diff
			}
			if diff < threshold {
				return true
			}
		}
	}
	return false
}
""",
        "wrong": """
	return false
}
"""
    },
    "r": {
        "correct": """
    n <- length(numbers)
    for (i in 1:(n-1)) {
        for (j in (i+1):n) {
            if (abs(numbers[i] - numbers[j]) < threshold) {
                return(TRUE)
            }
        }
    }
    return(FALSE)
}
""",
        "wrong": """
    return(FALSE)
}
"""
    },
}


def test_language(lang: str) -> bool:
    if lang not in TESTS:
        print(f"  {lang}: No test defined, skipping")
        return True

    dataset_map = {
        "js": "humaneval-js",
        "ts": "humaneval-ts",
        "rs": "humaneval-rs",
        "go": "humaneval-go",
        "lua": "humaneval-lua",
        "rb": "humaneval-rb",
        "cpp": "humaneval-cpp",
        "java": "humaneval-java",
        "php": "humaneval-php",
        "r": "humaneval-r",
        "pl": "humaneval-pl",
        "sh": "humaneval-sh",
    }

    ds = load_dataset("nuprl/MultiPL-E", dataset_map[lang], split="test")
    doc = ds[0]

    # Test correct solution
    program = build_program(doc["prompt"], TESTS[lang]["correct"], doc["tests"])
    result = eval_string_script(lang, program)
    correct_ok = result["status"] == "OK"

    # Test wrong solution
    program = build_program(doc["prompt"], TESTS[lang]["wrong"], doc["tests"])
    result = eval_string_script(lang, program)
    wrong_fails = result["status"] != "OK"

    passed = correct_ok and wrong_fails
    status = "PASS" if passed else "FAIL"
    detail = f"correct={'OK' if correct_ok else 'FAIL'}, wrong={'FAIL' if wrong_fails else 'OK(bad)'}"
    print(f"  {lang}: {status} ({detail})")

    if not passed and result.get("stderr"):
        print(f"      stderr: {result['stderr'][:200]}")

    return passed


def main():
    print("=" * 60)
    print("MultiPL-E Evaluation Test")
    print("=" * 60)

    # Skip languages with test issues:
    # - lua: needs luaunit
    # - java: needs org.javatuples
    # - pl: test harness issues with our test completion
    langs = ["js", "ts", "rs", "go", "rb", "cpp", "php", "r", "sh"]
    results = {}

    for lang in langs:
        try:
            results[lang] = test_language(lang)
        except Exception as e:
            print(f"  {lang}: ERROR - {e}")
            results[lang] = False

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"SUMMARY: {passed}/{total} languages passed")
    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
