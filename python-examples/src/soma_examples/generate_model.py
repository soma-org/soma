import asyncio

from soma_models.v1.flax import Model, ModelConfig


async def run():
    from flax import nnx

    model = Model(ModelConfig(dropout_rate=0.0), rngs=nnx.Rngs(0))
    model.save_bytes()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
