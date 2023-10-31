from argparse import ArgumentParser
from transformers.commands.convert import ConvertCommand
from transformers.commands.download import DownloadCommand
from transformers.commands.env import EnvironmentCommand
from transformers.commands.run import RunCommand
from transformers.commands.serving import ServeCommand
from transformers.commands.user import UserCommands

def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.commands.transformers_cli.main', 'main()', {'ArgumentParser': ArgumentParser, 'ConvertCommand': ConvertCommand, 'DownloadCommand': DownloadCommand, 'EnvironmentCommand': EnvironmentCommand, 'RunCommand': RunCommand, 'ServeCommand': ServeCommand, 'UserCommands': UserCommands}, 0)
if __name__ == '__main__':
    main()

