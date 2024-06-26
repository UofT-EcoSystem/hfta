from hfta.hfht.utils import resolve_overlap_runtimes, fuse_dicts

TESTS = []


def register_test(test_name):

  def test_decor(test_func):

    def _wrapper(*args, **kwargs):
      print("Running test {}".format(test_name))
      try:
        res = test_func(*args, **kwargs)
      except Exception as er:
        print("Exception occurred when running test {}\n{}".format(
            test_name, er))
        res = False
      if not res:
        print("FAILED")
      else:
        print("PASSED")

    TESTS.append(_wrapper)
    return _wrapper

  return test_decor


@register_test("resolve_overlap_runtimes")
def test_resolve_overlap_runtimes():
  tic_tocs = [
      [],
      [(1, 10)],
      [(1, 9), (2, 6)],
      [(2, 6), (1, 9)],
      [(1, 2), (2, 3), (3, 4), (10, 11)],
      [(1, 100), (20, 100), (60, 100), (90, 100)],
      [(3.3, 5.5), (7.7, 8.8)],
      [(3.3, 5.5), (3.3, 5.5)],
      [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100),
       (0, 100), (0, 100), (0, 100)],
      [(3.0, 5.0), (1.5, 3.5), (2.5, 4.0), (1, 3.5)],
  ]
  ans = [
      [],
      [9],
      [6.0, 2.0],
      [2.0, 6.0],
      [1.0, 1.0, 1.0, 1.0],
      [51.5, 32.5, 12.5, 2.5],
      [2.2, 1.1],
      [1.1, 1.1],
      [10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
      [
          0.5 / 4 + 0.5 / 2 + 1.0,
          1.0 / 2 + 0.5 / 3 + 0.5 / 4,
          0.5 / 3 + 0.5 / 4 + 0.5 / 2,
          0.5 + 1.0 / 2 + 0.5 / 3 + 0.5 / 4,
      ],
  ]
  correct = True
  for test_input, expected in zip(tic_tocs, ans):
    res = resolve_overlap_runtimes(test_input)
    if not all([abs(i - j) < 0.0001 for i, j in zip(res, expected)]):
      print("Expecting {} got {}".format(expected, res))
      correct = False
  return correct


@register_test('fuse_dicts')
def test_fuse_dicts():
  ins = [
      # Empty list.
      [],
      # Single element.
      [
          {
              'a': 1
          },
      ],
      [
          {
              'a': 1,
              'b': 10
          },
      ],
      # Multiple elements.
      [
          {
              'a': 1,
              'b': 10,
              'c': 100
          },
          {
              'a': 2,
              'b': 20,
              'c': 200
          },
          {
              'a': 3,
              'b': 30,
              'c': 300
          },
      ],
      # Multiple elements with missing values.
      [
          {
              'a': 1,
              'b': 10,
          },
          {
              'b': 20,
              'c': 200
          },
          {
              'a': 3,
              'c': 300
          },
      ],
  ]
  expected_outs = [
      {},
      {
          'a': [1]
      },
      {
          'a': [1],
          'b': [10]
      },
      {
          'a': [1, 2, 3],
          'b': [10, 20, 30],
          'c': [100, 200, 300]
      },
      {
          'a': [1, None, 3],
          'b': [10, 20, None],
          'c': [None, 200, 300]
      },
  ]

  def _assert_equal(a, b):
    assert len(a) == len(b)
    for ka, va in a.items():
      assert ka in b
      vb = b[ka]
      assert len(va) == len(vb)
      for ea, eb in zip(va, vb):
        assert ea == eb

  for inp, expected_out in zip(ins, expected_outs):
    _assert_equal(fuse_dicts(inp), expected_out)

  return True


if __name__ == "__main__":
  print("==" * 50)
  print()
  for func in TESTS:
    func()
