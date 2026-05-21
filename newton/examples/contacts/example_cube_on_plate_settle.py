# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import newton.examples
from newton.examples.contacts.example_cube_on_plate import SCENARIO_SETTLE
from newton.examples.contacts.example_cube_on_plate import Example as _Example


class Example(_Example):
    @staticmethod
    def create_parser():
        parser = _Example.create_parser()
        parser.set_defaults(scenario=SCENARIO_SETTLE)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
