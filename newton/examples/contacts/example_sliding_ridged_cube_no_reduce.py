# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import newton.examples
from newton.examples.contacts.example_sliding_ridged_cube import Example as _Example


class Example(_Example):
    @staticmethod
    def create_parser():
        parser = _Example.create_parser()
        parser.set_defaults(reduce_contacts=False)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
