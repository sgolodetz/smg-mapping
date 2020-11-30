from smg.mapping import MappingServer


def main() -> None:
    server: MappingServer = MappingServer()
    server.start()


if __name__ == "__main__":
    main()
