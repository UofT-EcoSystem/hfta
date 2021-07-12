def testcase_automator(testcase, configs):
  print('Running testcase: {} ...'.format(testcase.__name__))
  for name, vals in configs.items():
    print('\tTesting along {} ...'.format(name))
    for val in vals:
      print('\t\tTry {}={}'.format(name, val))
      kwargs = {name: val}
      testcase(**kwargs)
