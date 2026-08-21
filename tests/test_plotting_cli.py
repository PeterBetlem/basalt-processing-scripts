from basalt_processing.plotting import build_parser


def test_plotting_parser_accepts_output_only():
    args = build_parser().parse_args(["--output", "timeline.png"])
    assert str(args.output) == "timeline.png"
