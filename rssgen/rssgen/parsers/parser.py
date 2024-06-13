from rssgen.parsers.parser_factory import ParserFactory
from rssgen.utils import log


def configure_subparsers(subparsers) -> None:
    """Configure the subparsers for additional arguments."""
    pass


def parse_config(dataset, config_file):
    """Generate configuration parameters according to the passed yaml file"""
    parser = ParserFactory.create_parser(dataset, config_file)
    parsed_values = parser.parse()
    log("info", "Parsed values", parsed_values)
    return parsed_values
