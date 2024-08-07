from bindings import seldoncore

options = seldoncore.parse_config_file('/home/parrot_user/Desktop/pyseldon/subprojects/seldon/examples/ActivityDriven/conf.toml')

seldoncore.run_simulation(options = options)