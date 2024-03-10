#!/bin/sh

set -e

git config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
    while read path_key path
    do
        name=$(echo $path_key | sed 's/\submodule\.\(.*\)\.path/\1/')
        url_key=$(echo $path_key | sed 's/\.path/.url/')
        branch_key=$(echo $path_key | sed 's/\.path/.branch/')
        url=$(git config -f .gitmodules --get "$url_key")
        branch=$(git config -f .gitmodules --get "$branch_key" || echo "master")
        git submodule add -b $branch --name $name $url $path || continue
    done
