import pip
import getpass
import os

pkg = input("Enter Package you wish to install:");

def install(package):
    pip.main(["install",package,"--proxy=http://"+os.environ.get("USERNAME")+":"+getpass.getpass("Password for "+os.environ.get("USERNAME")+":")+"@proxy.ihc.com:8080"])

install(pkg)

