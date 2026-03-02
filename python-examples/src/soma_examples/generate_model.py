import asyncio

from soma_models.v1.flax import Model, ModelConfig


async def run():
    model = Model(ModelConfig(dropout_rate=0.0))
    model.save_bytes()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
