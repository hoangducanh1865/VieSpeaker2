import subprocess
Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
cmd = "gdown --id %s -O %s"%(Link, "pretrain_TalkSet.model")
subprocess.call(cmd, shell=True, stdout=None)