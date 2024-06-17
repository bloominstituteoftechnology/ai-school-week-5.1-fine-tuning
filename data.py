training_data= [
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef add(a, b):\n    return a + b\n```",
        "completion": "```python\nimport unittest\n\nclass TestAddFunction(unittest.TestCase):\n    def test_add_large_numbers(self):\n        self.assertEqual(add(1e10, 1e10), 2e10)\n\n    def test_add_float_numbers(self):\n        self.assertAlmostEqual(add(0.1, 0.2), 0.3, places=7)\n\n    def test_add_string_numbers(self):\n        with self.assertRaises(TypeError):\n            add('1', 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef multiply(a, b):\n    return a * b\n```",
        "completion": "```python\nimport unittest\n\nclass TestMultiplyFunction(unittest.TestCase):\n    def test_multiply_large_numbers(self):\n        self.assertEqual(multiply(1e10, 1e10), 1e20)\n\n    def test_multiply_by_float_numbers(self):\n        self.assertAlmostEqual(multiply(0.1, 0.2), 0.02, places=7)\n\n    def test_multiply_by_string_numbers(self):\n        with self.assertRaises(TypeError):\n            multiply('1', 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b\n```",
        "completion": "```python\nimport unittest\n\nclass TestDivideFunction(unittest.TestCase):\n    def test_divide_large_numbers(self):\n        self.assertAlmostEqual(divide(1e10, 1e2), 1e8)\n\n    def test_divide_by_float_numbers(self):\n        self.assertAlmostEqual(divide(0.1, 0.2), 0.5, places=7)\n\n    def test_divide_by_zero_float(self):\n        with self.assertRaises(ValueError):\n            divide(1.0, 0.0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef subtract(a, b):\n    return a - b\n```",
        "completion": "```python\nimport unittest\n\nclass TestSubtractFunction(unittest.TestCase):\n    def test_subtract_large_numbers(self):\n        self.assertEqual(subtract(1e10, 1e9), 9e9)\n\n    def test_subtract_float_numbers(self):\n        self.assertAlmostEqual(subtract(0.1, 0.2), -0.1, places=7)\n\n    def test_subtract_string_numbers(self):\n        with self.assertRaises(TypeError):\n            subtract('1', 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef power(a, b):\n    return a ** b\n```",
        "completion": "```python\nimport unittest\n\nclass TestPowerFunction(unittest.TestCase):\n    def test_power_large_numbers(self):\n        self.assertEqual(power(10, 10), 1e10)\n\n    def test_power_zero_exponent(self):\n        self.assertEqual(power(2, 0), 1)\n\n    def test_power_negative_exponent(self):\n        self.assertAlmostEqual(power(2, -1), 0.5, places=7)\n\n    def test_power_float_exponent(self):\n        self.assertAlmostEqual(power(9, 0.5), 3, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef modulo(a, b):\n    return a % b\n```",
        "completion": "```python\nimport unittest\n\nclass TestModuloFunction(unittest.TestCase):\n    def test_modulo_positive_numbers(self):\n        self.assertEqual(modulo(10, 3), 1)\n\n    def test_modulo_negative_numbers(self):\n        self.assertEqual(modulo(-10, 3), -1)\n\n    def test_modulo_by_zero(self):\n        with self.assertRaises(ZeroDivisionError):\n            modulo(10, 0)\n\n    def test_modulo_float_numbers(self):\n        self.assertAlmostEqual(modulo(10.5, 3.2), 0.9, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n```",
        "completion": "```python\nimport unittest\n\nclass TestFactorialFunction(unittest.TestCase):\n    def test_factorial_zero(self):\n        self.assertEqual(factorial(0), 1)\n\n    def test_factorial_positive_number(self):\n        self.assertEqual(factorial(5), 120)\n\n    def test_factorial_large_number(self):\n        self.assertEqual(factorial(10), 3628800)\n\n    def test_factorial_negative_number(self):\n        with self.assertRaises(RecursionError):\n            factorial(-1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n```",
        "completion": "```python\nimport unittest\n\nclass TestFibonacciFunction(unittest.TestCase):\n    def test_fibonacci_zero(self):\n        self.assertEqual(fibonacci(0), 0)\n\n    def test_fibonacci_one(self):\n        self.assertEqual(fibonacci(1), 1)\n\n    def test_fibonacci_positive_number(self):\n        self.assertEqual(fibonacci(5), 5)\n\n    def test_fibonacci_large_number(self):\n        self.assertEqual(fibonacci(10), 55)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsPrimeFunction(unittest.TestCase):\n    def test_is_prime_small_prime(self):\n        self.assertTrue(is_prime(7))\n\n    def test_is_prime_small_non_prime(self):\n        self.assertFalse(is_prime(4))\n\n    def test_is_prime_large_prime(self):\n        self.assertTrue(is_prime(29))\n\n    def test_is_prime_large_non_prime(self):\n        self.assertFalse(is_prime(30))\n\n    def test_is_prime_negative_number(self):\n        self.assertFalse(is_prime(-5))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef is_even(n):\n    return n % 2 == 0\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsEvenFunction(unittest.TestCase):\n    def test_is_even_positive_number(self):\n        self.assertTrue(is_even(4))\n\n    def test_is_even_negative_number(self):\n        self.assertTrue(is_even(-4))\n\n    def test_is_even_odd_number(self):\n        self.assertFalse(is_even(5))\n\n    def test_is_even_zero(self):\n        self.assertTrue(is_even(0))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsPalindromeFunction(unittest.TestCase):\n    def test_is_palindrome_even_length(self):\n        self.assertTrue(is_palindrome('abba'))\n\n    def test_is_palindrome_odd_length(self):\n        self.assertTrue(is_palindrome('racecar'))\n\n    def test_is_palindrome_not_palindrome(self):\n        self.assertFalse(is_palindrome('hello'))\n\n    def test_is_palindrome_empty_string(self):\n        self.assertTrue(is_palindrome(''))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef reverse_string(s):\n    return s[::-1]\n```",
        "completion": "```python\nimport unittest\n\nclass TestReverseStringFunction(unittest.TestCase):\n    def test_reverse_string_regular(self):\n        self.assertEqual(reverse_string('hello'), 'olleh')\n\n    def test_reverse_string_empty(self):\n        self.assertEqual(reverse_string(''), '')\n\n    def test_reverse_string_single_character(self):\n        self.assertEqual(reverse_string('a'), 'a')\n\n    def test_reverse_string_palindrome(self):\n        self.assertEqual(reverse_string('racecar'), 'racecar')\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef find_max(arr):\n    return max(arr)\n```",
        "completion": "```python\nimport unittest\n\nclass TestFindMaxFunction(unittest.TestCase):\n    def test_find_max_positive_numbers(self):\n        self.assertEqual(find_max([1, 2, 3, 4, 5]), 5)\n\n    def test_find_max_negative_numbers(self):\n        self.assertEqual(find_max([-1, -2, -3, -4, -5]), -1)\n\n    def test_find_max_mixed_numbers(self):\n        self.assertEqual(find_max([-1, 0, 1, 2, 3]), 3)\n\n    def test_find_max_single_element(self):\n        self.assertEqual(find_max([42]), 42)\n\n    def test_find_max_empty_list(self):\n        with self.assertRaises(ValueError):\n            find_max([])\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef find_min(arr):\n    return min(arr)\n```",
        "completion": "```python\nimport unittest\n\nclass TestFindMinFunction(unittest.TestCase):\n    def test_find_min_positive_numbers(self):\n        self.assertEqual(find_min([1, 2, 3, 4, 5]), 1)\n\n    def test_find_min_negative_numbers(self):\n        self.assertEqual(find_min([-1, -2, -3, -4, -5]), -5)\n\n    def test_find_min_mixed_numbers(self):\n        self.assertEqual(find_min([-1, 0, 1, 2, 3]), -1)\n\n    def test_find_min_single_element(self):\n        self.assertEqual(find_min([42]), 42)\n\n    def test_find_min_empty_list(self):\n        with self.assertRaises(ValueError):\n            find_min([])\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef sum_list(arr):\n    return sum(arr)\n```",
        "completion": "```python\nimport unittest\n\nclass TestSumListFunction(unittest.TestCase):\n    def test_sum_list_positive_numbers(self):\n        self.assertEqual(sum_list([1, 2, 3, 4, 5]), 15)\n\n    def test_sum_list_negative_numbers(self):\n        self.assertEqual(sum_list([-1, -2, -3, -4, -5]), -15)\n\n    def test_sum_list_mixed_numbers(self):\n        self.assertEqual(sum_list([-1, 0, 1, 2, 3]), 5)\n\n    def test_sum_list_empty_list(self):\n        self.assertEqual(sum_list([]), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef average_list(arr):\n    return sum(arr) / len(arr) if arr else 0\n```",
        "completion": "```python\nimport unittest\n\nclass TestAverageListFunction(unittest.TestCase):\n    def test_average_list_positive_numbers(self):\n        self.assertAlmostEqual(average_list([1, 2, 3, 4, 5]), 3.0, places=7)\n\n    def test_average_list_negative_numbers(self):\n        self.assertAlmostEqual(average_list([-1, -2, -3, -4, -5]), -3.0, places=7)\n\n    def test_average_list_mixed_numbers(self):\n        self.assertAlmostEqual(average_list([-1, 0, 1, 2, 3]), 1.0, places=7)\n\n    def test_average_list_empty_list(self):\n        self.assertEqual(average_list([]), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef count_vowels(s):\n    return sum(1 for char in s if char.lower() in 'aeiou')\n```",
        "completion": "```python\nimport unittest\n\nclass TestCountVowelsFunction(unittest.TestCase):\n    def test_count_vowels_all_vowels(self):\n        self.assertEqual(count_vowels('aeiou'), 5)\n\n    def test_count_vowels_mixed_characters(self):\n        self.assertEqual(count_vowels('hello world'), 3)\n\n    def test_count_vowels_no_vowels(self):\n        self.assertEqual(count_vowels('bcdfg'), 0)\n\n    def test_count_vowels_empty_string(self):\n        self.assertEqual(count_vowels(''), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef count_consonants(s):\n    return sum(1 for char in s if char.lower() in 'bcdfghjklmnpqrstvwxyz')\n```",
        "completion": "```python\nimport unittest\n\nclass TestCountConsonantsFunction(unittest.TestCase):\n    def test_count_consonants_all_consonants(self):\n        self.assertEqual(count_consonants('bcdfg'), 5)\n\n    def test_count_consonants_mixed_characters(self):\n        self.assertEqual(count_consonants('hello world'), 7)\n\n    def test_count_consonants_no_consonants(self):\n        self.assertEqual(count_consonants('aeiou'), 0)\n\n    def test_count_consonants_empty_string(self):\n        self.assertEqual(count_consonants(''), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef is_anagram(s1, s2):\n    return sorted(s1) == sorted(s2)\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsAnagramFunction(unittest.TestCase):\n    def test_is_anagram_positive(self):\n        self.assertTrue(is_anagram('listen', 'silent'))\n\n    def test_is_anagram_negative(self):\n        self.assertFalse(is_anagram('hello', 'world'))\n\n    def test_is_anagram_empty_strings(self):\n        self.assertTrue(is_anagram('', ''))\n\n    def test_is_anagram_case_sensitive(self):\n        self.assertFalse(is_anagram('Listen', 'Silent'))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef is_unique_chars(s):\n    return len(s) == len(set(s))\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsUniqueCharsFunction(unittest.TestCase):\n    def test_is_unique_chars_unique(self):\n        self.assertTrue(is_unique_chars('abcdef'))\n\n    def test_is_unique_chars_non_unique(self):\n        self.assertFalse(is_unique_chars('hello'))\n\n    def test_is_unique_chars_empty_string(self):\n        self.assertTrue(is_unique_chars(''))\n\n    def test_is_unique_chars_single_character(self):\n        self.assertTrue(is_unique_chars('a'))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef is_sorted(arr):\n    return arr == sorted(arr)\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsSortedFunction(unittest.TestCase):\n    def test_is_sorted_sorted_list(self):\n        self.assertTrue(is_sorted([1, 2, 3, 4, 5]))\n\n    def test_is_sorted_unsorted_list(self):\n        self.assertFalse(is_sorted([5, 3, 2, 4, 1]))\n\n    def test_is_sorted_single_element(self):\n        self.assertTrue(is_sorted([42]))\n\n    def test_is_sorted_empty_list(self):\n        self.assertTrue(is_sorted([]))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef merge_dicts(d1, d2):\n    result = d1.copy()\n    result.update(d2)\n    return result\n```",
        "completion": "```python\nimport unittest\n\nclass TestMergeDictsFunction(unittest.TestCase):\n    def test_merge_dicts_non_overlapping_keys(self):\n        self.assertEqual(merge_dicts({'a': 1}, {'b': 2}), {'a': 1, 'b': 2})\n\n    def test_merge_dicts_overlapping_keys(self):\n        self.assertEqual(merge_dicts({'a': 1}, {'a': 2}), {'a': 2})\n\n    def test_merge_dicts_empty_dicts(self):\n        self.assertEqual(merge_dicts({}, {}), {})\n\n    def test_merge_dicts_one_empty_dict(self):\n        self.assertEqual(merge_dicts({'a': 1}, {}), {'a': 1})\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef flatten_list(nested_list):\n    return [item for sublist in nested_list for item in sublist]\n```",
        "completion": "```python\nimport unittest\n\nclass TestFlattenListFunction(unittest.TestCase):\n    def test_flatten_list_single_level(self):\n        self.assertEqual(flatten_list([[1, 2], [3, 4]]), [1, 2, 3, 4])\n\n    def test_flatten_list_nested(self):\n        self.assertEqual(flatten_list([[1, [2]], [3, 4]]), [1, [2], 3, 4])\n\n    def test_flatten_list_empty(self):\n        self.assertEqual(flatten_list([]), [])\n\n    def test_flatten_list_mixed_types(self):\n        self.assertEqual(flatten_list([[1, 'a'], [3, 4.0]]), [1, 'a', 3, 4.0])\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    }
]

validation_data= [
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef add(a, b):\n    return a + b\n```",
        "completion": "```python\nimport unittest\n\nclass TestAddFunction(unittest.TestCase):\n    def test_add_positive_numbers(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_add_negative_numbers(self):\n        self.assertEqual(add(-1, -1), -2)\n\n    def test_add_positive_and_negative_numbers(self):\n        self.assertEqual(add(-1, 1), 0)\n\n    def test_add_zero(self):\n        self.assertEqual(add(0, 0), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef multiply(a, b):\n    return a * b\n```",
        "completion": "```python\nimport unittest\n\nclass TestMultiplyFunction(unittest.TestCase):\n    def test_multiply_positive_numbers(self):\n        self.assertEqual(multiply(2, 3), 6)\n\n    def test_multiply_negative_numbers(self):\n        self.assertEqual(multiply(-1, -1), 1)\n\n    def test_multiply_positive_and_negative_numbers(self):\n        self.assertEqual(multiply(-1, 1), -1)\n\n    def test_multiply_by_zero(self):\n        self.assertEqual(multiply(0, 5), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b\n```",
        "completion": "```python\nimport unittest\n\nclass TestDivideFunction(unittest.TestCase):\n    def test_divide_positive_numbers(self):\n        self.assertEqual(divide(6, 3), 2)\n\n    def test_divide_negative_numbers(self):\n        self.assertEqual(divide(-6, -3), 2)\n\n    def test_divide_positive_and_negative_numbers(self):\n        self.assertEqual(divide(-6, 3), -2)\n\n    def test_divide_by_zero(self):\n        with self.assertRaises(ValueError):\n            divide(6, 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "Generate a unit test for this function: \n```python\ndef subtract(a, b):\n    return a - b\n```",
        "completion": "```python\nimport unittest\n\nclass TestSubtractFunction(unittest.TestCase):\n    def test_subtract_positive_numbers(self):\n        self.assertEqual(subtract(5, 3), 2)\n\n    def test_subtract_negative_numbers(self):\n        self.assertEqual(subtract(-5, -3), -2)\n\n    def test_subtract_positive_and_negative_numbers(self):\n        self.assertEqual(subtract(-5, 3), -8)\n\n    def test_subtract_zero(self):\n        self.assertEqual(subtract(0, 0), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    }
]
